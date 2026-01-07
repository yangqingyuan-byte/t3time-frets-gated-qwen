import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import h5py
from torch.utils.data import DataLoader

# 确保项目根目录在 python 路径中
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models.T3Time_Wavelet_Decomp_Gated_Qwen import TriModalWaveletDecompGatedQwen
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from utils.experiment_logger import log_experiment_result

def infer_d_llm(args):
    path = f"./Embeddings/{args.data_path}/{args.embed_version}/train/0.h5"
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            data = f['embeddings'][:]
            d_llm = data.shape[0] if len(data.shape) == 2 else data.shape[1]
            print(f"[Info] 自动探测 d_llm: {d_llm}")
            return d_llm
    return args.d_llm

class CorrelationLoss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred_mean = torch.mean(y_pred, dim=1, keepdim=True)
        y_true_mean = torch.mean(y_true, dim=1, keepdim=True)
        pred_centered = y_pred - y_pred_mean
        true_centered = y_true - y_true_mean
        corr = torch.sum(pred_centered * true_centered, dim=1) / (
            torch.sqrt(torch.sum(pred_centered**2, dim=1) * torch.sum(true_centered**2, dim=1) + 1e-8)
        )
        return 1 - torch.mean(corr)

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.d_llm = infer_d_llm(args)
    
    # Dataset selection
    data_dict = {"ETTh1": Dataset_ETT_hour, "ETTh2": Dataset_ETT_hour, "ETTm1": Dataset_ETT_minute, "ETTm2": Dataset_ETT_minute}
    DataClass = data_dict.get(args.data_path, Dataset_Custom)
    
    train_set = DataClass(flag='train', size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    test_set = DataClass(flag='test', size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    model = TriModalWaveletDecompGatedQwen(
        device=device, channel=args.channel, num_nodes=args.num_nodes,
        seq_len=args.seq_len, pred_len=args.pred_len, dropout_n=args.dropout_n,
        d_llm=args.d_llm, wavelet=args.wavelet
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    mse_criterion = nn.MSELoss()
    shape_criterion = CorrelationLoss()
    
    best_mse = float('inf')
    best_mae = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        for i, (input, real, mark, mark_y, embeddings) in enumerate(train_loader):
            input, real, embeddings = input.float().to(device), real.float().to(device), embeddings.float().to(device)
            optimizer.zero_grad()
            output = model(input, mark, embeddings)
            
            mse_loss = mse_criterion(output, real)
            shape_loss = shape_criterion(output, real)
            loss = mse_loss + args.shape_lambda * shape_loss
            
            loss.backward()
            optimizer.step()
            
        # Test
        model.eval()
        mses, maes = [], []
        with torch.no_grad():
            for input, real, mark, mark_y, embeddings in test_loader:
                input, real, embeddings = input.to(device), real.to(device), embeddings.to(device)
                output = model(input, mark, embeddings)
                mses.append(nn.MSELoss()(output, real).item())
                maes.append(nn.L1Loss()(output, real).item())
        
        curr_mse, curr_mae = np.mean(mses), np.mean(maes)
        if curr_mse < best_mse:
            best_mse, best_mae = curr_mse, curr_mae
            
        print(f"Epoch {epoch+1}: Test MSE={curr_mse:.6f}, MAE={curr_mae:.6f}")
        
    log_experiment_result(
        args.data_path, args.pred_len, "Wavelet_Decomp_Gated_Qwen", args.seed,
        best_mse, best_mae, embed_version=args.embed_version, channel=args.channel,
        learning_rate=args.learning_rate, dropout_n=args.dropout_n
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_n", type=float, default=0.2)
    parser.add_argument("--channel", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--embed_version", type=str, default="qwen3_0.6b")
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--shape_lambda", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    train(args)

