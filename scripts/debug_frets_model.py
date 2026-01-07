"""
T3Time_FreTS_Gated_Qwen 模型诊断工具
用于系统性分析模型性能问题
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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour
from models.T3Time_FreTS_Gated_Qwen import TriModalFreTSGatedQwen
from models.T3Time_Learnable_Wavelet_Packet_Gated_Pro_Qwen import TriModalLearnableWaveletPacketGatedProQwen
from utils.metrics import metric

def analyze_gradient_flow(model, loss):
    """分析梯度流，检查是否存在梯度消失或爆炸"""
    avg_grads = []
    max_grads = []
    layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            avg_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())
    
    return {
        'layers': layers,
        'avg_grads': avg_grads,
        'max_grads': max_grads
    }

def analyze_feature_statistics(model, data_loader, device):
    """分析模型各层的特征统计信息"""
    model.eval()
    stats = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                stats[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'shape': list(output.shape)
                }
        return hook
    
    # 注册 hooks
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # 运行一个 batch
    with torch.no_grad():
        for bx, by, bxm, bym, emb in data_loader:
            bx, by = bx.to(device).float(), by.to(device).float()
            emb = emb.to(device).float()
            _ = model(bx, None, emb)
            break
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    return stats

def compare_models_outputs(frets_model, t3time_model, data_loader, device):
    """对比两个模型的输出差异"""
    frets_model.eval()
    t3time_model.eval()
    
    frets_preds = []
    t3time_preds = []
    trues = []
    
    with torch.no_grad():
        for bx, by, bxm, bym, emb in data_loader:
            bx, by = bx.to(device).float(), by.to(device).float()
            emb = emb.to(device).float()
            
            frets_out = frets_model(bx, None, emb)
            t3time_out = t3time_model(bx, bxm, emb)
            
            frets_preds.append(frets_out.cpu())
            t3time_preds.append(t3time_out.cpu())
            trues.append(by[:, -96:, :].cpu())
    
    frets_preds = torch.cat(frets_preds, dim=0)
    t3time_preds = torch.cat(t3time_preds, dim=0)
    trues = torch.cat(trues, dim=0)
    
    # 计算指标
    frets_mse, frets_mae = metric(frets_preds, trues)
    t3time_mse, t3time_mae = metric(t3time_preds, trues)
    
    # 计算输出差异
    output_diff = (frets_preds - t3time_preds).abs()
    
    return {
        'frets_mse': frets_mse,
        'frets_mae': frets_mae,
        't3time_mse': t3time_mse,
        't3time_mae': t3time_mae,
        'output_diff_mean': output_diff.mean().item(),
        'output_diff_max': output_diff.max().item(),
        'output_diff_std': output_diff.std().item()
    }

def analyze_frequency_domain(frets_model, data_loader, device):
    """分析频域分支的输出"""
    frets_model.eval()
    
    # 提取频域特征
    frequency_features = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if 'frets_branch' in name or 'fre_encoder' in name:
                frequency_features.append({
                    'name': name,
                    'output': output.detach().cpu() if isinstance(output, torch.Tensor) else None
                })
        return hook
    
    hooks = []
    for name, module in frets_model.named_modules():
        if 'frets' in name.lower() or 'fre' in name.lower():
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    with torch.no_grad():
        for bx, by, bxm, bym, emb in data_loader:
            bx = bx.to(device).float()
            emb = emb.to(device).float()
            _ = frets_model(bx, None, emb)
            break
    
    for hook in hooks:
        hook.remove()
    
    return frequency_features

def diagnose_fusion_mechanism(frets_model, data_loader, device):
    """诊断融合机制是否正常工作"""
    frets_model.eval()
    
    time_features = []
    freq_features = []
    fused_features = []
    
    def hook_time(module, input, output):
        if isinstance(output, torch.Tensor):
            time_features.append(output.detach().cpu())
        elif isinstance(output, tuple):
            time_features.append(output[0].detach().cpu() if isinstance(output[0], torch.Tensor) else None)
    
    def hook_freq(module, input, output):
        if isinstance(output, torch.Tensor):
            freq_features.append(output.detach().cpu())
        elif isinstance(output, tuple):
            freq_features.append(output[0].detach().cpu() if isinstance(output[0], torch.Tensor) else None)
    
    def hook_fusion(module, input, output):
        if isinstance(output, tuple):
            fused_features.append(output[0].detach().cpu() if isinstance(output[0], torch.Tensor) else None)
        elif isinstance(output, torch.Tensor):
            fused_features.append(output.detach().cpu())
    
    # 注册 hooks
    hooks = []
    for name, module in frets_model.named_modules():
        if 'ts_encoder' in name and len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_time))
        elif 'fre_encoder' in name:
            hooks.append(module.register_forward_hook(hook_freq))
        elif 'cross_attn_fusion' in name:
            hooks.append(module.register_forward_hook(hook_fusion))
    
    with torch.no_grad():
        for bx, by, bxm, bym, emb in data_loader:
            bx = bx.to(device).float()
            emb = emb.to(device).float()
            _ = frets_model(bx, None, emb)
            break
    
    for hook in hooks:
        hook.remove()
    
    return {
        'time_feature_norm': [f.norm().item() for f in time_features] if time_features else [],
        'freq_feature_norm': [f.norm().item() for f in freq_features] if freq_features else [],
        'fused_feature_norm': [f.norm().item() for f in fused_features] if fused_features else []
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    data_set = Dataset_ETT_hour(
        root_path='./dataset/',
        data_path='ETTh1.csv',
        flag='test',
        size=[96, 0, 96],
        features='M',
        embed_version='qwen3_0.6b'
    )
    data_loader = DataLoader(data_set, batch_size=16, shuffle=False, num_workers=4, drop_last=True)
    
    print("="*80)
    print("T3Time_FreTS_Gated_Qwen 模型诊断报告")
    print("="*80)
    
    # 1. 初始化模型
    print("\n[1] 初始化模型...")
    frets_model = TriModalFreTSGatedQwen(
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
    
    t3time_model = TriModalLearnableWaveletPacketGatedProQwen(
        device=device,
        channel=128,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.5,
        wp_level=2
    ).to(device)
    
    print(f"FreTS 模型参数量: {sum(p.numel() for p in frets_model.parameters() if p.requires_grad):,}")
    print(f"T3Time 模型参数量: {sum(p.numel() for p in t3time_model.parameters() if p.requires_grad):,}")
    
    # 2. 对比模型输出
    print("\n[2] 对比模型输出...")
    comparison = compare_models_outputs(frets_model, t3time_model, data_loader, device)
    print(f"FreTS 模型 - MSE: {comparison['frets_mse']:.6f}, MAE: {comparison['frets_mae']:.6f}")
    print(f"T3Time 模型 - MSE: {comparison['t3time_mse']:.6f}, MAE: {comparison['t3time_mae']:.6f}")
    print(f"输出差异 - Mean: {comparison['output_diff_mean']:.6f}, Max: {comparison['output_diff_max']:.6f}, Std: {comparison['output_diff_std']:.6f}")
    
    # 3. 分析特征统计
    print("\n[3] 分析特征统计...")
    stats = analyze_feature_statistics(frets_model, data_loader, device)
    print("关键层特征统计:")
    for name, stat in list(stats.items())[:10]:
        print(f"  {name}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, range=[{stat['min']:.4f}, {stat['max']:.4f}]")
    
    # 4. 诊断融合机制
    print("\n[4] 诊断融合机制...")
    fusion_diag = diagnose_fusion_mechanism(frets_model, data_loader, device)
    if fusion_diag['time_feature_norm']:
        print(f"时域特征范数: {fusion_diag['time_feature_norm']}")
    if fusion_diag['freq_feature_norm']:
        print(f"频域特征范数: {fusion_diag['freq_feature_norm']}")
    if fusion_diag['fused_feature_norm']:
        print(f"融合特征范数: {fusion_diag['fused_feature_norm']}")
    
    # 5. 分析频域分支
    print("\n[5] 分析频域分支...")
    freq_features = analyze_frequency_domain(frets_model, data_loader, device)
    print(f"频域分支层数: {len(freq_features)}")
    for feat in freq_features[:5]:
        if feat['output'] is not None:
            print(f"  {feat['name']}: shape={feat['output'].shape if hasattr(feat['output'], 'shape') else 'N/A'}")
    
    print("\n" + "="*80)
    print("诊断完成！")
    print("="*80)

if __name__ == "__main__":
    main()
