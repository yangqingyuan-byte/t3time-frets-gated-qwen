"""
训练带 Gated Attention 和形态损失 (Correlation Loss) 的 T3Time 模型
"""
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import argparse
import time
import os
import random
import h5py
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.T3Time_Wavelet_Gated_Shape_Qwen import TriModalWaveletGatedShapeQwen
from utils.metrics import MSE, MAE, metric
from utils.experiment_logger import log_experiment_result

def correlation_loss(y_pred, y_true):
    """
    计算相关性损失: 1 - Pearson相关系数
    y_pred, y_true: [B, L, N]
    """
    y_pred = y_pred.float()
    y_true = y_true.float()
    # 减去均值
    y_pred_mean = y_pred.mean(dim=1, keepdim=True)
    y_true_mean = y_true.mean(dim=1, keepdim=True)
    y_pred_cent = y_pred - y_pred_mean
    y_true_cent = y_true - y_true_mean
    
    # 计算协方差和标准差
    cov = (y_pred_cent * y_true_cent).sum(dim=1)
    std_pred = torch.sqrt((y_pred_cent ** 2).sum(dim=1) + 1e-8)
    std_true = torch.sqrt((y_true_cent ** 2).sum(dim=1) + 1e-8)
    
    corr = cov / (std_pred * std_true)
    return 1 - corr.mean()

def infer_d_llm(args):
    # 修正探测路径
    path = f"./Embeddings/{args.data_path}/{args.embed_version}/train/0.h5"
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            data = f['embeddings'][:]
            d_llm = data.shape[0] if len(data.shape) == 2 else data.shape[1]
            print(f"[Info] 自动探测 d_llm: {d_llm}")
            return d_llm
    return args.d_llm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--channel", type=int, default=64)
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_n", type=float, default=0.3)
    parser.add_argument("--d_llm", type=int, default=1024)
    parser.add_argument("--e_layer", type=int, default=1)
    parser.add_argument("--d_layer", type=int, default=1)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100) # 统一改为 100
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument("--es_patience", type=int, default=25)
    parser.add_argument("--shape_lambda", type=float, default=0.1, help="形态损失权重")
    parser.add_argument("--embed_version", type=str, default="qwen3_0.6b")
    parser.add_argument("--save_model", action="store_true", default=False)
    return parser.parse_args()

class trainer:
    def __init__(self, args):
        self.model = TriModalWaveletGatedShapeQwen(
            device=args.device, channel=args.channel, num_nodes=args.num_nodes, 
            seq_len=args.seq_len, pred_len=args.pred_len, dropout_n=args.dropout_n, 
            d_llm=args.d_llm, e_layer=args.e_layer, d_layer=args.d_layer, head=args.head
        ).to(args.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        self.shape_lambda = args.shape_lambda

    def train(self, x, y, x_mark, emb):
        self.model.train()
        self.optimizer.zero_grad()
        # 强制转换为 float 防止 Double 错误
        x, y, emb = x.float(), y.float(), emb.float()
        pred = self.model(x, x_mark, emb).float()
        mse_loss = MSE(pred, y)
        s_loss = correlation_loss(pred, y)
        total_loss = mse_loss + self.shape_lambda * s_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        return mse_loss.item(), MAE(pred, y).item()

    def eval(self, x, y, x_mark, emb):
        self.model.eval()
        with torch.no_grad():
            x, y, emb = x.float(), y.float(), emb.float()
            pred = self.model(x, x_mark, emb).float()
            return MSE(pred, y).item(), MAE(pred, y).item()

def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    
    args.d_llm = infer_d_llm(args)

    data_map = {'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour, 'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute}
    data_class = data_map.get(args.data_path, Dataset_Custom)
    train_set = data_class(flag='train', size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    val_set = data_class(flag='val', size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    test_set = data_class(flag='test', size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    engine = trainer(args)
    best_val_loss = float('inf'); best_model_state = None; patience = 0

    for epoch in range(args.epochs):
        t = time.time()
        for x, y, x_mark, y_mark, emb in train_loader:
            engine.train(x.to(args.device), y.to(args.device), x_mark.to(args.device), emb.to(args.device))
        
        val_losses = []
        for x, y, x_mark, y_mark, emb in val_loader:
            v_mse, _ = engine.eval(x.to(args.device), y.to(args.device), x_mark.to(args.device), emb.to(args.device))
            val_losses.append(v_mse)
        
        m_val = np.mean(val_losses)
        print(f"Epoch {epoch+1}: Val MSE {m_val:.4f}, Time {time.time()-t:.2f}s")
        
        if m_val < best_val_loss:
            best_val_loss = m_val; patience = 0
            best_model_state = {k: v.cpu() for k, v in engine.model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.es_patience: break
        engine.scheduler.step()

    if best_model_state:
        engine.model.load_state_dict(best_model_state)
    
    test_mse, test_mae = [], []
    for x, y, x_mark, y_mark, emb in test_loader:
        mse, mae = engine.eval(x.to(args.device), y.to(args.device), x_mark.to(args.device), emb.to(args.device))
        test_mse.append(mse); test_mae.append(mae)
    
    f_mse, f_mae = np.mean(test_mse), np.mean(test_mae)
    print(f"\nFinal Test MSE: {f_mse:.6f}, Test MAE: {f_mae:.6f}")

    log_experiment_result(
        data_path=args.data_path, pred_len=args.pred_len, model_name="T3Time_Wavelet_Gated_Shape_Qwen",
        seed=args.seed, test_mse=f_mse, test_mae=f_mae, embed_version=args.embed_version,
        seq_len=args.seq_len, channel=args.channel, batch_size=args.batch_size,
        learning_rate=args.learning_rate, dropout_n=args.dropout_n,
        additional_info={"shape_lambda": args.shape_lambda}
    )

if __name__ == "__main__":
    main()
