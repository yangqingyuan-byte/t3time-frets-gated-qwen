import torch
import torch.nn as nn
from torch import optim
import numpy as np
import argparse
import os
import random
import h5py
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.T3Time_Learnable_Wavelet_Gated_Shape_Qwen import TriModalLearnableWaveletGatedShapeQwen
from utils.experiment_logger import log_experiment_result

def correlation_loss(y_pred, y_true):
    y_pred, y_true = y_pred.float(), y_true.float()
    y_pred_mean, y_true_mean = y_pred.mean(dim=1, keepdim=True), y_true.mean(dim=1, keepdim=True)
    y_pred_cent, y_true_cent = y_pred - y_pred_mean, y_true - y_true_mean
    cov = (y_pred_cent * y_true_cent).sum(dim=1)
    std_pred = torch.sqrt((y_pred_cent ** 2).sum(dim=1) + 1e-8)
    std_true = torch.sqrt((y_true_cent ** 2).sum(dim=1) + 1e-8)
    corr = cov / (std_pred * std_true)
    return 1 - corr.mean()

def infer_d_llm(args):
    path = f"./Embeddings/{args.data_path}/{args.embed_version}/train/0.h5"
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            data = f['embeddings'][:]
            return data.shape[0] if len(data.shape) == 2 else data.shape[1]
    return args.d_llm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--channel", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument('--seed', type=int, default=2031)
    parser.add_argument("--shape_lambda", type=float, default=0.1)
    parser.add_argument("--levels", type=int, default=3, help="小波分解层级")
    parser.add_argument("--embed_version", type=str, default="qwen3_0.6b")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.d_llm = infer_d_llm(args)

    data_map = {'ETTh1': Dataset_ETT_hour, 'ETTm1': Dataset_ETT_minute}
    data_class = data_map.get(args.data_path, Dataset_Custom)
    train_set = data_class(flag='train', size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    test_set = data_class(flag='test', size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = TriModalLearnableWaveletGatedShapeQwen(
        device=args.device, channel=args.channel, seq_len=args.seq_len, 
        pred_len=args.pred_len, d_llm=args.d_llm, levels=args.levels
    ).to(args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = torch.nn.MSELoss()

    best_mse, best_mae = float('inf'), float('inf')

    for epoch in range(args.epochs):
        model.train()
        for x, y, x_mark, y_mark, emb in train_loader:
            x, y, emb = x.float().to(args.device), y.float().to(args.device), emb.float().to(args.device)
            optimizer.zero_grad()
            pred = model(x, x_mark, emb).float()
            loss = criterion(pred, y) + args.shape_lambda * correlation_loss(pred, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        model.eval()
        mses, maes = [], []
        with torch.no_grad():
            for x, y, x_mark, y_mark, emb in test_loader:
                x, y, emb = x.float().to(args.device), y.float().to(args.device), emb.float().to(args.device)
                pred = model(x, x_mark, emb).float()
                mses.append(criterion(pred, y).item())
                maes.append(torch.abs(pred - y).mean().item())
        
        curr_mse = np.mean(mses)
        if curr_mse < best_mse:
            best_mse, best_mae = curr_mse, np.mean(maes)
        print(f"Epoch {epoch+1}: Test MSE {curr_mse:.6f}")

    log_experiment_result(
        args.data_path, args.pred_len, "T3Time_Learnable_Wavelet_Gated_Shape_Qwen",
        args.seed, best_mse, best_mae, embed_version=args.embed_version,
        channel=args.channel, learning_rate=args.learning_rate,
        additional_info={"learnable_wavelet": True, "levels": args.levels}
    )

if __name__ == "__main__":
    main()

