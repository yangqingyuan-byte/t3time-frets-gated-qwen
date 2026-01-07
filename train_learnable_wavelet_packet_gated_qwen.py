import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import time
from datetime import datetime
import json

from models.T3Time_Learnable_Wavelet_Packet_Gated_Qwen import TriModalLearnableWaveletPacketGatedQwen
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_Custom
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

def main():
    parser = argparse.ArgumentParser(description='T3Time Learnable Wavelet Packet Gated Qwen Training')

    # Data loading
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--num_nodes', type=int, default=7, help='number of nodes')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    
    # Model parameters
    parser.add_argument('--channel', type=int, default=32, help='channel dimension')
    parser.add_argument('--dropout_n', type=float, default=0.1, help='dropout')
    parser.add_argument('--e_layer', type=int, default=1, help='encoder layers')
    parser.add_argument('--d_layer', type=int, default=1, help='decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='feedforward dimension')
    parser.add_argument('--head', type=int, default=8, help='attention heads')
    parser.add_argument('--wp_level', type=int, default=2, help='wavelet packet level')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--embed_version', type=str, default='qwen3_0.6b', help='embedding version')

    args = parser.parse_args()
    
    # 随机种子设定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径和节点数自动纠正
    if args.data_path == 'ETTh1': 
        args.data_path = 'ETTh1.csv'
        args.num_nodes = 7
    elif args.data_path == 'ETTh2': 
        args.data_path = 'ETTh2.csv'
        args.num_nodes = 7
    
    # 确保 checkpoints 目录存在
    if not os.path.exists("./checkpoints/"):
        os.makedirs("./checkpoints/")

    # Data provider
    def data_provider(args, flag):
        Data = Dataset_ETT_hour if 'ETT' in args.data_path else Dataset_Custom
        shuffle_flag = True if flag == 'train' else False
        
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, 0, args.pred_len],
            features='M',
            embed_version=args.embed_version
        )
        dataloader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=4,
            drop_last=True
        )
        return data_set, dataloader

    train_data, train_loader = data_provider(args, 'train')
    val_data, val_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # Model
    model = TriModalLearnableWaveletPacketGatedQwen(
        device=device,
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        d_ff=args.d_ff,
        head=args.head,
        wp_level=args.wp_level
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # Training
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, embeddings) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            batch_x_mark = batch_x_mark.to(device).float()
            embeddings = embeddings.to(device).float()

            outputs = model(batch_x, batch_x_mark, embeddings)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()

        train_loss = np.average(train_loss)
        
        # Validation
        model.eval()
        val_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, embeddings) in enumerate(val_loader):
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()
                batch_x_mark = batch_x_mark.to(device).float()
                embeddings = embeddings.to(device).float()
                outputs = model(batch_x, batch_x_mark, embeddings)
                loss = criterion(outputs, batch_y)
                val_loss.append(loss.item())
        val_loss = np.average(val_loss)

        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        early_stopping(val_loss, model, "./checkpoints/")
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        adjust_learning_rate(optimizer, epoch + 1, args)

    # Testing
    model.load_state_dict(torch.load("./checkpoints/checkpoint.pth"))
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, embeddings) in enumerate(test_loader):
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            batch_x_mark = batch_x_mark.to(device).float()
            embeddings = embeddings.to(device).float()
            outputs = model(batch_x, batch_x_mark, embeddings)
            preds.append(outputs.detach().cpu())
            trues.append(batch_y.detach().cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    
    # 计算指标
    mse, mae = metric(preds, trues)
    print(f'Test Results: MSE: {mse:.6f}, MAE: {mae:.6f}')

    # 记录到主日志
    log_file = "/root/0/T3Time/experiment_results.log"
    with open(log_file, "a") as f:
        res = {
            "data_path": args.data_path.replace(".csv", ""),
            "pred_len": args.pred_len,
            "test_mse": float(mse),
            "test_mae": float(mae),
            "model": "T3Time_Learnable_Wavelet_Packet",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": args.seed,
            "seq_len": args.seq_len,
            "channel": args.channel,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout_n": args.dropout_n,
            "wp_level": args.wp_level
        }
        f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()
