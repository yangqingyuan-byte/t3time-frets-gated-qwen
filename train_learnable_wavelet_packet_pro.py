import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import json

from models.T3Time_Learnable_Wavelet_Packet_Gated_Pro_Qwen import TriModalLearnableWaveletPacketGatedProQwen
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_Custom
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

# 【V18/V27】 Shape Loss 定义
def correlation_loss(pred, target):
    pred_mean = pred - pred.mean(dim=1, keepdim=True)
    target_mean = target - target.mean(dim=1, keepdim=True)
    pred_std = pred_mean.norm(dim=1, keepdim=True) + 1e-8
    target_std = target_mean.norm(dim=1, keepdim=True) + 1e-8
    corr = (pred_mean * target_mean).sum(dim=1, keepdim=True) / (pred_std * target_std)
    return (1 - corr).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--model_id', type=str, default='T3Time_Pro_Qwen', help='Identifier for the run')
    parser.add_argument('--model', type=str, default='TriModalLearnableWaveletPacketGatedProQwen', help='Model name')
    parser.add_argument('--num_nodes', type=int, default=7)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--channel', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--dropout_n', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Balanced L2 regularization')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lradj', type=str, default='type1', help='using balanced decay')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--wp_level', type=int, default=2, help='Level 2 for better frequency resolution')
    parser.add_argument('--embed_version', type=str, default='qwen3_0.6b')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设备选择
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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

    model = TriModalLearnableWaveletPacketGatedProQwen(
        device=device, 
        channel=args.channel, 
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len, 
        dropout_n=args.dropout_n,
        wp_level=args.wp_level
    ).to(device)

    # V25 配置: 纯净 MSE Loss
    criterion = nn.MSELoss()
    
    # V25 配置: Adam 优化器 + Step Decay
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for i, (bx, by, bxm, bym, emb) in enumerate(train_loader):
            optimizer.zero_grad()
            bx, by = bx.to(device).float(), by.to(device).float()
            bxm, emb = bxm.to(device).float(), emb.to(device).float()
            
            outputs = model(bx, bxm, emb)
            
            # 简单纯粹的 MSE
            mse_loss = criterion(outputs, by)
            
            # 不加 Shape Loss，避免干扰 MSE 的极值搜索
            l1_lambda = 1e-4
            sparsity_loss = model.get_sparsity_loss()
            loss = mse_loss + l1_lambda * sparsity_loss
            
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # V25 配置: Step Decay
        adjust_learning_rate(optimizer, epoch + 1, args)
        
        model.eval()
        val_loss = []
        with torch.no_grad():
            for bx, by, bxm, bym, emb in val_loader:
                bx, by = bx.to(device).float(), by.to(device).float()
                bxm, emb = bxm.to(device).float(), emb.to(device).float()
                outputs = model(bx, bxm, emb)
                val_loss.append(criterion(outputs, by).item())
        
        v_loss = np.average(val_loss)
        print(f"Epoch {epoch+1}, LR: {optimizer.param_groups[0]['lr']:.8f}, Train Loss: {np.average(train_loss):.6f}, Val Loss: {v_loss:.6f}")
        
        early_stopping(v_loss, model, "./checkpoints/")
        if early_stopping.early_stop: break

    # 【V19 集成支持】 根据 seed 保存不同名称的 checkpoint
    checkpoint_path = f"./checkpoints/checkpoint_seed_{args.seed}.pth"
    if os.path.exists("./checkpoints/checkpoint.pth"):
        os.rename("./checkpoints/checkpoint.pth", checkpoint_path)
        print(f"Checkpoint saved as: {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by, bxm, bym, emb in test_loader:
            bx, by = bx.to(device).float(), by.to(device).float()
            bxm, emb = bxm.to(device).float(), emb.to(device).float()
            outputs = model(bx, bxm, emb)
            preds.append(outputs.detach().cpu()); trues.append(by.detach().cpu())

    preds, trues = torch.cat(preds, dim=0), torch.cat(trues, dim=0)
    mse, mae = metric(preds, trues)
    print(f'Pro Test Results: MSE: {mse:.6f}, MAE: {mae:.6f}')

    with open("/root/0/T3Time/experiment_results.log", "a") as f:
        res = {
            "model_id": args.model_id,
            "data_path": args.data_path.replace(".csv", ""), 
            "pred_len": args.pred_len, 
            "test_mse": float(mse), 
            "test_mae": float(mae), 
            "model": args.model, 
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": args.seed,
            "seq_len": args.seq_len,
            "channel": args.channel,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout_n": args.dropout_n,
            "weight_decay": args.weight_decay,
            "wp_level": args.wp_level
        }
        f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()
