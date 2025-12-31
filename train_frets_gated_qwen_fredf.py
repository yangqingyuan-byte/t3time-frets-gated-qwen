import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
import os
import random
import json
import sys

# 将 plug_and_play 添加到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'plug_and_play'))

from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.T3Time_FreTS_Gated_Qwen_FreDF import TriModalFreTSGatedQwenFreDF
from models.T3Time_FreTS_Gated_Qwen import TriModalFreTSGatedQwen
from models.T3Time_FFT_Qwen import TriModalFFTQwen
from utils.metrics import metric
from fredf_loss import FreDFLoss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='FreTS_FreDF', help='FreTS_FreDF, FreTS_Only, FFT_FreDF')
    parser.add_argument('--data_path', type=str, default='ETTh1')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--num_nodes', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--dropout_n', type=float, default=0.1)
    parser.add_argument('--channel', type=int, default=64)
    parser.add_argument('--e_layer', type=int, default=1)
    parser.add_argument('--d_layer', type=int, default=1)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--es_patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--embed_version', type=str, default='qwen3_0.6b')
    parser.add_argument('--lradj', type=str, default='type1')
    # FreDF 特定参数
    parser.add_argument('--lambda_freq', type=float, default=0.5, help='频率损失权重')
    parser.add_argument('--freq_mode', type=str, default='rfft', help='rfft 或 fft')
    return parser.parse_args()

def data_provider(args, flag):
    Data = Dataset_ETT_hour
    if args.data_path.startswith('ETTm'): Data = Dataset_ETT_minute
    elif args.data_path not in ['ETTh1', 'ETTh2']: Data = Dataset_Custom
    
    data_file = args.data_path
    if not data_file.endswith('.csv') and data_file not in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        data_file += '.csv'
    
    # 特殊处理 ETT 系列，Dataset_ETT_hour 期望 'ETTh1.csv' 而非 'ETTh1'
    if data_file == 'ETTh1': data_file = 'ETTh1.csv'
    if data_file == 'ETTh2': data_file = 'ETTh2.csv'
    if data_file == 'ETTm1': data_file = 'ETTm1.csv'
    if data_file == 'ETTm2': data_file = 'ETTm2.csv'

    shuffle = (flag != 'test')
    drop_last = True

    dataset = Data(
        root_path='./dataset/',
        data_path=data_file,
        flag=flag,
        size=[args.seq_len, 0, args.pred_len],
        features='M',
        scale=True,
        embed_version=args.embed_version
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=4,
        drop_last=drop_last
    )
    return dataset, dataloader

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset, train_loader = data_provider(args, 'train')
    val_dataset, val_loader = data_provider(args, 'val')
    _, test_loader = data_provider(args, 'test')

    if args.model == 'FreTS_FreDF':
        model_class = TriModalFreTSGatedQwenFreDF
    elif args.model == 'FreTS_Only':
        model_class = TriModalFreTSGatedQwen
    elif args.model == 'FFT_FreDF':
        model_class = TriModalFFTQwen
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model_class(
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
    
    # 使用 FreDF 损失函数
    criterion = FreDFLoss(lambda_freq=args.lambda_freq, loss_type='MSE', freq_mode=args.freq_mode)
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, embeddings) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x, batch_y, embeddings = batch_x.to(device).float(), batch_y.to(device).float(), embeddings.to(device).float()
            
            outputs = model(batch_x, batch_x_mark, embeddings)
            
            # 使用 FreDF 计算总损失
            loss, loss_tmp, loss_freq = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        val_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, embeddings) in enumerate(val_loader):
                batch_x, batch_y, embeddings = batch_x.to(device).float(), batch_y.to(device).float(), embeddings.to(device).float()
                outputs = model(batch_x, batch_x_mark, embeddings)
                loss, _, _ = criterion(outputs, batch_y)
                val_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        val_loss = np.mean(val_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoint_fredf.pth')
        else:
            patience_counter += 1
            if patience_counter >= args.es_patience:
                print("Early stopping")
                break
        
        adjust_learning_rate(optimizer, epoch + 1, args)

    model.load_state_dict(torch.load('checkpoint_fredf.pth'))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, embeddings) in enumerate(test_loader):
            batch_x, batch_y, embeddings = batch_x.to(device).float(), batch_y.to(device).float(), embeddings.to(device).float()
            outputs = model(batch_x, batch_x_mark, embeddings)
            preds.append(outputs.detach().cpu())
            trues.append(batch_y.detach().cpu())

    preds, trues = torch.cat(preds, dim=0), torch.cat(trues, dim=0)
    mse, mae = metric(preds, trues)
    print(f"On average horizons, Test MSE: {mse:.6f}, Test MAE: {mae:.6f}")

if __name__ == "__main__":
    main()

