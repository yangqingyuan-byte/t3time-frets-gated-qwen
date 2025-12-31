import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.T3Time_FreTS_Gated_Qwen import TriModalFreTSGatedQwen
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import numpy as np
import time
import json
import random

def data_provider(args, flag):
    Data = Dataset_ETT_hour
    if args.data_path.startswith('ETTm'): Data = Dataset_ETT_minute
    elif args.data_path not in ['ETTh1', 'ETTh2']: Data = Dataset_Custom
    
    data_file = args.data_path
    if not data_file.endswith('.csv') and data_file not in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        data_file += '.csv'

    data_set = Data(
        root_path='./dataset/',
        data_path=data_file,
        flag=flag,
        size=[args.seq_len, 0, args.pred_len],
        features='M',
        num_nodes=args.num_nodes,
        scale=True, # 必须设为 True 以在归一化空间稳定训练
        embed_version=args.embed_version
    )
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=(flag != 'test'), num_workers=4, drop_last=True)
    return data_set, data_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='ETTh1')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--num_nodes', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--dropout_n', type=float, default=0.1)
    parser.add_argument('--channel', type=int, default=64)
    parser.add_argument('--e_layer', type=int, default=1)
    parser.add_argument('--d_layer', type=int, default=1)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--es_patience', type=int, default=10)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--embed_version', type=str, default='qwen3_0.6b')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, data_loader = data_provider(args, 'train')
    _, vali_loader = data_provider(args, 'val')
    _, test_loader = data_provider(args, 'test')

    model = TriModalFreTSGatedQwen(
        device=device, 
        channel=args.channel, 
        num_nodes=args.num_nodes, 
        seq_len=args.seq_len, 
        pred_len=args.pred_len, 
        dropout_n=args.dropout_n,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.es_patience, verbose=True)

    print(f"Final Standard Training Started - Pred_Len: {args.pred_len}")

    for epoch in range(args.epochs):
        model.train(); train_loss = []
        for i, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            batch_x, batch_y, be = batch_data[0].to(device).float(), batch_data[1].to(device).float(), batch_data[-1].to(device).float()
            outputs = model(batch_x, None, be)
            batch_y_pred = batch_y[:, -args.pred_len:, :]
            loss = criterion(outputs, batch_y_pred)
            train_loss.append(loss.item()); loss.backward(); optimizer.step()
        
        model.eval(); vali_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                bx, by, bee = batch_data[0].to(device).float(), batch_data[1].to(device).float(), batch_data[-1].to(device).float()
                out = model(bx, None, bee)
                by_pred = by[:, -args.pred_len:, :]
                vali_loss.append(criterion(out, by_pred).item())
        
        avg_vali_loss = np.mean(vali_loss)
        print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_loss):.6f} Vali Loss: {avg_vali_loss:.6f}")
        early_stopping(avg_vali_loss, model, "./")
        if early_stopping.early_stop: break
        adjust_learning_rate(optimizer, epoch + 1, args)

    model.load_state_dict(torch.load('checkpoint.pth')); model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            bx, by, bee = batch_data[0].to(device).float(), batch_data[1].to(device).float(), batch_data[-1].to(device).float()
            out = model(bx, None, bee)
            by_pred = by[:, -args.pred_len:, :]
            preds.append(out.detach().cpu()); trues.append(by_pred.detach().cpu())

    preds, trues = torch.cat(preds, dim=0), torch.cat(trues, dim=0)
    mse, mae = metric(preds, trues)
    print(f"On average horizons, Test MSE: {mse:.6f}, Test MAE: {mae:.6f}")

    # 注释掉内部日志记录，由外部 shell 脚本统一处理，以对齐官方风格
    # log_record = {"data_path": args.data_path, ...}
    # with open("experiment_results.log", "a") as f: f.write(json.dumps(log_record) + "\n")

if __name__ == "__main__": main()
