"""
快速诊断 T3Time_FreTS_Gated_Qwen 模型
简化版本，快速定位问题
"""
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour
from models.T3Time_FreTS_Gated_Qwen import TriModalFreTSGatedQwen
from utils.metrics import metric

def quick_diagnose():
    """快速诊断模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载测试数据
    data_set = Dataset_ETT_hour(
        root_path='./dataset/',
        data_path='ETTh1.csv',
        flag='test',
        size=[96, 0, 96],
        features='M',
        embed_version='qwen3_0.6b'
    )
    data_loader = DataLoader(data_set, batch_size=16, shuffle=False, num_workers=4, drop_last=True)
    
    model = TriModalFreTSGatedQwen(
        device=device,
        channel=64,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.1,
        e_layer=1,
        d_layer=1,
        head=8
    ).to(device)
    
    model.eval()
    
    print("="*60)
    print("快速诊断报告")
    print("="*60)
    
    # 1. 检查模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[1] 模型参数量: {total_params:,}")
    
    # 2. 前向传播测试
    print("\n[2] 前向传播测试...")
    try:
        with torch.no_grad():
            for bx, by, bxm, bym, emb in data_loader:
                bx, by = bx.to(device).float(), by.to(device).float()
                emb = emb.to(device).float()
                
                outputs = model(bx, None, emb)
                by_pred = by[:, -96:, :]
                
                mse, mae = metric(outputs.cpu(), by_pred.cpu())
                
                print(f"  ✅ 前向传播成功")
                print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")
                print(f"  输出形状: {outputs.shape}")
                print(f"  输出范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
                print(f"  输出均值: {outputs.mean():.4f}, 标准差: {outputs.std():.4f}")
                break
    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
        return
    
    # 3. 检查关键组件
    print("\n[3] 检查关键组件...")
    
    # 检查归一化层
    norm_weight = model.normalize_layers.affine_weight if hasattr(model.normalize_layers, 'affine_weight') else None
    if norm_weight is not None:
        print(f"  RevIN affine_weight: mean={norm_weight.mean():.4f}, std={norm_weight.std():.4f}")
    
    # 检查频域分支
    frets_params = sum(p.numel() for p in model.frets_branch.parameters())
    print(f"  FreTS 分支参数量: {frets_params:,}")
    
    # 检查融合层
    fusion_params = sum(p.numel() for p in model.cross_attn_fusion.parameters())
    print(f"  Cross-Attention 融合参数量: {fusion_params:,}")
    
    # 4. 梯度测试
    print("\n[4] 梯度测试...")
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    try:
        for bx, by, bxm, bym, emb in data_loader:
            bx, by = bx.to(device).float(), by.to(device).float()
            emb = emb.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(bx, None, emb)
            by_pred = by[:, -96:, :]
            loss = criterion(outputs, by_pred)
            loss.backward()
            
            # 检查梯度
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    if grad_norm > 100:
                        print(f"  ⚠️  梯度爆炸: {name}, norm={grad_norm:.4f}")
                    if grad_norm < 1e-6:
                        print(f"  ⚠️  梯度消失: {name}, norm={grad_norm:.4f}")
            
            if grad_norms:
                print(f"  梯度范数统计: mean={np.mean(grad_norms):.4f}, max={np.max(grad_norms):.4f}, min={np.min(grad_norms):.4f}")
            print(f"  ✅ 梯度计算成功, Loss: {loss.item():.6f}")
            break
    except Exception as e:
        print(f"  ❌ 梯度计算失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("诊断完成！")
    print("="*60)
    print("\n💡 建议:")
    print("  1. 如果 MSE/MAE 异常高，检查数据归一化")
    print("  2. 如果梯度异常，检查学习率和初始化")
    print("  3. 如果输出范围异常，检查 RevIN 归一化")
    print("  4. 运行完整诊断: python scripts/debug_frets_model.py")
    print("  5. 查看调试指南: cat scripts/DEBUG_GUIDE_FreTS.md")

if __name__ == "__main__":
    quick_diagnose()
