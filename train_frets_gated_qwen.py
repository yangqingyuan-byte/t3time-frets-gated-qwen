"""
T3Time_FreTS_Gated_Qwen 训练脚本
使用最佳配置的简化版本（不带消融选项）
"""
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
from datetime import datetime

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
        scale=True,
        embed_version=args.embed_version
    )
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=(flag != 'test'), num_workers=0, drop_last=True)
    return data_set, data_loader

def main():
    parser = argparse.ArgumentParser(description='T3Time_FreTS_Gated_Qwen 训练脚本（最佳配置）')
    parser.add_argument('--data_path', type=str, default='ETTh1', help='数据集路径')
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测长度')
    parser.add_argument('--num_nodes', type=int, default=7, help='节点数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--dropout_n', type=float, default=0.1, help='Dropout 比率')
    parser.add_argument('--channel', type=int, default=64, help='通道数')
    parser.add_argument('--e_layer', type=int, default=1, help='编码器层数')
    parser.add_argument('--d_layer', type=int, default=1, help='解码器层数')
    parser.add_argument('--head', type=int, default=8, help='注意力头数')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--es_patience', type=int, default=10, help='Early stopping 耐心值')
    parser.add_argument('--lradj', type=str, default='type1', help='学习率调整策略: type1 (Step Decay)')
    parser.add_argument('--embed_version', type=str, default='qwen3_0.6b', help='嵌入版本')
    parser.add_argument('--seed', type=int, default=2024, help='随机种子')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--loss_fn', type=str, default='smooth_l1', choices=['mse', 'smooth_l1'],
                       help='损失函数: mse (MSELoss) or smooth_l1 (SmoothL1Loss)')
    parser.add_argument('--model_id', type=str, default='T3Time_FreTS_Gated_Qwen', help='模型标识符')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 清理可能存在的 checkpoint 文件
    checkpoint_files = ['./checkpoint.pth', './checkpoints/checkpoint.pth']
    for cf in checkpoint_files:
        if os.path.exists(cf):
            os.remove(cf)
            print(f"已清理旧的 checkpoint 文件: {cf}")

    # 加载数据
    _, data_loader = data_provider(args, 'train')
    _, vali_loader = data_provider(args, 'val')
    _, test_loader = data_provider(args, 'test')

    # 创建模型（使用最佳配置，不带消融选项）
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
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.loss_fn == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.SmoothL1Loss(beta=0.2)
    early_stopping = EarlyStopping(patience=args.es_patience, verbose=True)
    
    # 在内存中保存最佳模型状态（不保存到磁盘）
    best_model_state = None
    best_vali_loss = float('inf')

    print("="*60)
    print(f"训练开始 - T3Time_FreTS_Gated_Qwen (最佳配置)")
    print(f"Pred_Len: {args.pred_len}, Channel: {args.channel}")
    print(f"固定配置: FreTS Component, 稀疏化(0.009), 改进门控, Gate融合")
    print("="*60)

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for i, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            batch_x, batch_y, be = batch_data[0].to(device).float(), batch_data[1].to(device).float(), batch_data[-1].to(device).float()
            outputs = model(batch_x, None, be)
            batch_y_pred = batch_y[:, -args.pred_len:, :]
            loss = criterion(outputs, batch_y_pred)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        model.eval()
        vali_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                bx, by, bee = batch_data[0].to(device).float(), batch_data[1].to(device).float(), batch_data[-1].to(device).float()
                out = model(bx, None, bee)
                by_pred = by[:, -args.pred_len:, :]
                vali_loss.append(criterion(out, by_pred).item())
        
        avg_vali_loss = np.mean(vali_loss)
        print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_loss):.6f} Vali Loss: {avg_vali_loss:.6f}")
        
        # 保存最佳模型状态到内存（不保存到磁盘）
        if avg_vali_loss < best_vali_loss:
            best_vali_loss = avg_vali_loss
            best_model_state = model.state_dict().copy()
            print(f'Validation loss decreased to {best_vali_loss:.6f}. 保存最佳模型状态到内存（不保存文件）')
        
        # Early stopping 检查
        early_stopping(avg_vali_loss, model, "./")
        if early_stopping.early_stop: 
            break
        
        # 学习率调整
        adjust_learning_rate(optimizer, epoch + 1, args)

    # 加载内存中的最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ 已加载内存中的最佳模型状态（验证损失: {best_vali_loss:.6f}）")
    else:
        print("⚠️  未找到最佳模型状态，使用当前模型状态进行测试")
    
    # 测试
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            bx, by, bee = batch_data[0].to(device).float(), batch_data[1].to(device).float(), batch_data[-1].to(device).float()
            out = model(bx, None, bee)
            by_pred = by[:, -args.pred_len:, :]
            preds.append(out.detach().cpu())
            trues.append(by_pred.detach().cpu())

    preds, trues = torch.cat(preds, dim=0), torch.cat(trues, dim=0)
    mse, mae = metric(preds, trues)
    print(f"On average horizons, Test MSE: {mse:.6f}, Test MAE: {mae:.6f}")

    # 写入实验日志
    log_file = "/root/0/T3Time/experiment_results.log"
    with open(log_file, "a") as f:
        res = {
            "model_id": args.model_id,
            "data_path": args.data_path.replace(".csv", "") if args.data_path.endswith(".csv") else args.data_path,
            "pred_len": args.pred_len,
            "test_mse": float(mse),
            "test_mae": float(mae),
            "model": "T3Time_FreTS_Gated_Qwen",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": args.seed,
            "seq_len": args.seq_len,
            "channel": args.channel,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout_n": args.dropout_n,
            "weight_decay": args.weight_decay,
            "e_layer": args.e_layer,
            "d_layer": args.d_layer,
            "head": args.head,
            "embed_version": args.embed_version,
            "loss_fn": args.loss_fn,
            "lradj": args.lradj,
            "patience": args.es_patience,
            # 固定配置参数（用于记录）
            "sparsity_threshold": 0.009,
            "frets_scale": 0.018,
            "fusion_mode": "gate",
            "use_frets": True,
            "use_sparsity": True,
            "use_improved_gate": True
        }
        f.write(json.dumps(res) + "\n")
    print(f"✅ 实验结果已保存到 {log_file}")

if __name__ == "__main__": 
    main()
