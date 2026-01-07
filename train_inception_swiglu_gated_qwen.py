"""
训练带 Inception, SwiGLU, Gated Attention 的 T3Time 模型
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
from models.T3Time_Inception_SwiGLU_Gated_Qwen import TriModalInceptionSwiGLUGatedQwen
from utils.metrics import MSE, MAE
from utils.experiment_logger import log_experiment_result

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
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument("--es_patience", type=int, default=25)
    parser.add_argument("--embed_version", type=str, default="qwen3_0.6b")
    parser.add_argument("--save_model", action="store_true", default=False)
    return parser.parse_args()

class trainer:
    def __init__(self, args):
        self.model = TriModalInceptionSwiGLUGatedQwen(
            device=args.device, channel=args.channel, num_nodes=args.num_nodes, seq_len=args.seq_len,
            pred_len=args.pred_len, dropout_n=args.dropout_n, d_llm=args.d_llm, e_layer=args.e_layer,
            d_layer=args.d_layer, d_ff=args.d_ff, head=args.head
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)

    def train_eval_step(self, loader, engine_method, device):
        losses, maes = [], []
        for x, y, x_mark, y_mark, emb in loader:
            x, y, x_mark, emb = x.to(device), y.to(device), x_mark.to(device), emb.to(device)
            if engine_method == "train":
                self.model.train(); self.optimizer.zero_grad()
                pred = self.model(x, x_mark, emb)
                loss = MSE(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                losses.append(loss.item()); maes.append(MAE(pred, y).item())
            else:
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(x, x_mark, emb)
                    losses.append(MSE(pred, y).item()); maes.append(MAE(pred, y).item())
        return np.mean(losses), np.mean(maes)

def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    
    # 探测 d_llm
    embed_dir = f"./embeddings/{args.data_path}_{args.embed_version}/"
    if os.path.exists(embed_dir):
        fname = sorted(f for f in os.listdir(embed_dir) if f.endswith(".h5"))[0]
        with h5py.File(os.path.join(embed_dir, fname), 'r') as f:
            d = f['data']
            args.d_llm = d.shape[0] if len(d.shape)==2 else d.shape[1]

    data_map = {'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour, 'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute}
    data_class = data_map.get(args.data_path, Dataset_Custom)
    train_loader = DataLoader(data_class(flag='train', size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(data_class(flag='val', size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(data_class(flag='test', size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version), batch_size=args.batch_size, shuffle=False)

    engine = trainer(args)
    best_val_loss = float('inf'); best_model_state = None; patience = 0

    for epoch in range(args.epochs):
        t = time.time()
        train_mse, _ = engine.train_eval_step(train_loader, "train", args.device)
        val_mse, _ = engine.train_eval_step(val_loader, "eval", args.device)
        print(f"Epoch {epoch+1}: Train MSE {train_mse:.4f}, Val MSE {val_mse:.4f}, Time {time.time()-t:.2f}s")
        
        if val_mse < best_val_loss:
            best_val_loss = val_mse; patience = 0
            best_model_state = {k: v.cpu() for k, v in engine.model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.es_patience: break
        engine.scheduler.step()

    engine.model.load_state_dict(best_model_state)
    test_mse, test_mae = engine.train_eval_step(test_loader, "eval", args.device)
    print(f"\nFinal Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}")

    log_experiment_result(
        data_path=args.data_path, pred_len=args.pred_len, model_name="T3Time_Inception_SwiGLU_Gated_Qwen",
        seed=args.seed, test_mse=test_mse, test_mae=test_mae, embed_version=args.embed_version,
        seq_len=args.seq_len, channel=args.channel, batch_size=args.batch_size,
        learning_rate=args.learning_rate, dropout_n=args.dropout_n,
        additional_info={"inception": True, "swiglu": True}
    )

if __name__ == "__main__":
    main()

