"""
训练带 Gated Attention 和小波包分解 (WPD) 的 T3Time 模型
"""
import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
import h5py
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.T3Time_Wavelet_Packet_Gated_Qwen import TriModalWaveletPacketGatedQwen
from utils.metrics import MSE, MAE, metric
from utils.experiment_logger import log_experiment_result
import faulthandler
faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

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
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument("--es_patience", type=int, default=25)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wp_level", type=int, default=2, help="WPD level (2->4 nodes, 3->8 nodes)")
    parser.add_argument("--save", type=str, default="./logs_wavelet_packet/")
    parser.add_argument("--embed_version", type=str, default="qwen3_0.6b")
    parser.add_argument("--save_model", action="store_true", default=False)
    return parser.parse_args()

class trainer:
    def __init__(self, args):
        self.model = TriModalWaveletPacketGatedQwen(
            device=args.device, channel=args.channel, num_nodes=args.num_nodes, seq_len=args.seq_len,
            pred_len=args.pred_len, dropout_n=args.dropout_n, d_llm=args.d_llm, e_layer=args.e_layer,
            d_layer=args.d_layer, head=args.head, wavelet=args.wavelet, wp_level=args.wp_level
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        self.loss_fn = MSE
        self.mae_fn = MAE
    
    def train(self, x, y, x_mark, emb):
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(x, x_mark, emb)
        loss = self.loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        return loss.item(), self.mae_fn(pred, y).item()

    def eval(self, x, y, x_mark, emb):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x, x_mark, emb)
            return self.loss_fn(pred, y).item(), self.mae_fn(pred, y).item()

def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    
    # 自动探测 d_llm
    embed_dir = f"./embeddings/{args.data_path}_{args.embed_version}/"
    if os.path.exists(embed_dir):
        fname = sorted(f for f in os.listdir(embed_dir) if f.endswith(".h5"))[0]
        with h5py.File(os.path.join(embed_dir, fname), 'r') as f:
            d = f['data']
            args.d_llm = d.shape[0] if len(d.shape)==2 else d.shape[1]
            print(f"Detected d_llm: {args.d_llm}")

    data_map = {'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour, 'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute}
    data_class = data_map.get(args.data_path, Dataset_Custom)
    
    train_set = data_class(flag='train', size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    val_set = data_class(flag='val', size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    test_set = data_class(flag='test', size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    engine = trainer(args)
    best_val_loss = float('inf'); best_test_mse = 0; best_test_mae = 0
    patience_cnt = 0; best_model_state = None

    for epoch in range(args.epochs):
        t1 = time.time()
        train_loss = []
        for x, y, x_mark, y_mark, emb in train_loader:
            loss, _ = engine.train(x.to(args.device), y.to(args.device), x_mark.to(args.device), emb.to(args.device))
            train_loss.append(loss)
        
        val_loss = []
        for x, y, x_mark, y_mark, emb in val_loader:
            loss, _ = engine.eval(x.to(args.device), y.to(args.device), x_mark.to(args.device), emb.to(args.device))
            val_loss.append(loss)
        
        m_val_loss = np.mean(val_loss)
        print(f"Epoch {epoch+1}: Train {np.mean(train_loss):.4f}, Val {m_val_loss:.4f}, Time {time.time()-t1:.2f}s")
        
        if m_val_loss < best_val_loss:
            best_val_loss = m_val_loss; patience_cnt = 0
            best_model_state = {k: v.cpu() for k, v in engine.model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= args.es_patience: break
        engine.scheduler.step()

    # 测试
    engine.model.load_state_dict(best_model_state)
    test_mse, test_mae = [], []
    for x, y, x_mark, y_mark, emb in test_loader:
        mse, mae = engine.eval(x.to(args.device), y.to(args.device), x_mark.to(args.device), emb.to(args.device))
        test_mse.append(mse); test_mae.append(mae)
    
    final_mse, final_mae = np.mean(test_mse), np.mean(test_mae)
    print(f"\nFinal Test MSE: {final_mse:.6f}, Test MAE: {final_mae:.6f}")

    log_experiment_result(
        data_path=args.data_path,
        pred_len=args.pred_len,
        model_name="T3Time_Wavelet_Packet_Gated_Qwen",
        seed=args.seed,
        test_mse=final_mse,
        test_mae=final_mae,
        embed_version=args.embed_version,
        seq_len=args.seq_len,
        channel=args.channel,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_n=args.dropout_n,
        additional_info={"wp_level": args.wp_level}
    )

if __name__ == "__main__":
    main()

