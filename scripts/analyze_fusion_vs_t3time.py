"""
对比分析 T3Time_FreTS_FusionExp 与 T3Time_Pro_Qwen_SOTA_V30 的性能差异
系统性诊断为什么融合机制实验版本不如原始模型
"""
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour
from models.T3Time_FreTS_Gated_Qwen_FusionExp import TriModalFreTSGatedQwenFusionExp
from models.T3Time_Learnable_Wavelet_Packet_Gated_Pro_Qwen import TriModalLearnableWaveletPacketGatedProQwen
from utils.metrics import metric

def load_best_results():
    """从日志中加载最佳结果"""
    log_file = "/root/0/T3Time/experiment_results.log"
    results = {}
    
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                model_id = data.get('model_id', '')
                if 'T3Time_Pro_Qwen_SOTA_V30' in model_id and data.get('pred_len') == 96:
                    key = f"{model_id}_seed{data.get('seed', 'unknown')}"
                    if key not in results or data.get('test_mse', float('inf')) < results[key].get('test_mse', float('inf')):
                        results[key] = data
            except:
                continue
    
    return results

def compare_architectures():
    """对比两个模型的架构差异"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    frets_model = TriModalFreTSGatedQwenFusionExp(
        device=device,
        channel=64,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        fusion_mode='gate'
    ).to(device)
    
    t3time_model = TriModalLearnableWaveletPacketGatedProQwen(
        device=device,
        channel=128,  # T3Time V30 使用 128
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.5,  # T3Time V30 使用 0.5
        wp_level=2
    ).to(device)
    
    print("="*80)
    print("架构对比分析")
    print("="*80)
    
    # 1. 参数量对比
    frets_params = sum(p.numel() for p in frets_model.parameters() if p.requires_grad)
    t3time_params = sum(p.numel() for p in t3time_model.parameters() if p.requires_grad)
    
    print(f"\n[1] 参数量对比:")
    print(f"  FreTS FusionExp: {frets_params:,}")
    print(f"  T3Time V30:      {t3time_params:,}")
    print(f"  差异: {t3time_params - frets_params:,} ({((t3time_params - frets_params) / frets_params * 100):.1f}%)")
    
    # 2. 频域处理方式对比
    print(f"\n[2] 频域处理方式:")
    print(f"  FreTS: FreTS Component (单频域表示)")
    print(f"  T3Time: Learnable Wavelet Packet (4个频带，先验引导初始化)")
    
    # 3. 融合机制对比
    print(f"\n[3] 融合机制:")
    print(f"  FreTS: Gate/Weighted/Cross-Attn/Hybrid (实验版本)")
    print(f"  T3Time: Static Weights + Horizon-Aware Gate (V30)")
    
    # 4. 超参数对比
    print(f"\n[4] 关键超参数:")
    print(f"  FreTS: channel=64, dropout=0.1, weight_decay=1e-4")
    print(f"  T3Time: channel=128, dropout=0.5, weight_decay=1e-3")
    
    # 5. 正则化强度对比
    print(f"\n[5] 正则化强度:")
    frets_reg = 0.1 + 1e-4  # dropout + weight_decay
    t3time_reg = 0.5 + 1e-3
    print(f"  FreTS: {frets_reg:.4f} (较低)")
    print(f"  T3Time: {t3time_reg:.4f} (较高)")
    print(f"  差异: T3Time 使用了更强的正则化")
    
    return frets_model, t3time_model

def analyze_feature_statistics(frets_model, t3time_model, data_loader, device):
    """分析特征统计信息"""
    print("\n" + "="*80)
    print("特征统计对比")
    print("="*80)
    
    frets_model.eval()
    t3time_model.eval()
    
    frets_features = {
        'time': [], 'freq': [], 'fused': []
    }
    t3time_features = {
        'time': [], 'freq': [], 'fused': []
    }
    
    # Hook 函数
    def hook_frets_time(module, input, output):
        if isinstance(output, torch.Tensor):
            frets_features['time'].append(output.detach().cpu())
    
    def hook_frets_freq(module, input, output):
        if isinstance(output, torch.Tensor):
            frets_features['freq'].append(output.detach().cpu())
    
    def hook_t3time_time(module, input, output):
        if isinstance(output, torch.Tensor):
            t3time_features['time'].append(output.detach().cpu())
    
    def hook_t3time_freq(module, input, output):
        if isinstance(output, torch.Tensor):
            t3time_features['freq'].append(output.detach().cpu())
    
    # 注册 hooks
    hooks = []
    for name, module in frets_model.named_modules():
        if 'ts_encoder' in name and len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_frets_time))
        elif 'fre_encoder' in name:
            hooks.append(module.register_forward_hook(hook_frets_freq))
    
    for name, module in t3time_model.named_modules():
        if 'ts_encoder' in name and len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_t3time_time))
        elif 'wp_processing' in name or 'cf_interaction' in name:
            hooks.append(module.register_forward_hook(hook_t3time_freq))
    
    # 前向传播
    with torch.no_grad():
        for bx, by, bxm, bym, emb in data_loader:
            bx = bx.to(device).float()
            emb = emb.to(device).float()
            
            _ = frets_model(bx, None, emb)
            _ = t3time_model(bx, None, emb)
            break
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    # 分析统计信息
    def analyze_features(name, features):
        if not features:
            return None
        stacked = torch.cat(features, dim=0)
        return {
            'mean': stacked.mean().item(),
            'std': stacked.std().item(),
            'min': stacked.min().item(),
            'max': stacked.max().item(),
            'norm': stacked.norm().item() / stacked.numel()
        }
    
    print("\n[1] 时域特征统计:")
    frets_time_stat = analyze_features('FreTS Time', frets_features['time'])
    t3time_time_stat = analyze_features('T3Time Time', t3time_features['time'])
    if frets_time_stat and t3time_time_stat:
        print(f"  FreTS: mean={frets_time_stat['mean']:.4f}, std={frets_time_stat['std']:.4f}, norm={frets_time_stat['norm']:.4f}")
        print(f"  T3Time: mean={t3time_time_stat['mean']:.4f}, std={t3time_time_stat['std']:.4f}, norm={t3time_time_stat['norm']:.4f}")
    
    print("\n[2] 频域特征统计:")
    frets_freq_stat = analyze_features('FreTS Freq', frets_features['freq'])
    t3time_freq_stat = analyze_features('T3Time Freq', t3time_features['freq'])
    if frets_freq_stat and t3time_freq_stat:
        print(f"  FreTS: mean={frets_freq_stat['mean']:.4f}, std={frets_freq_stat['std']:.4f}, norm={frets_freq_stat['norm']:.4f}")
        print(f"  T3Time: mean={t3time_freq_stat['mean']:.4f}, std={t3time_freq_stat['std']:.4f}, norm={t3time_freq_stat['norm']:.4f}")
        if frets_freq_stat['norm'] < t3time_freq_stat['norm'] * 0.5:
            print(f"  ⚠️  警告: FreTS 频域特征范数明显小于 T3Time，可能信息丢失")

def analyze_output_distribution(frets_model, t3time_model, data_loader, device):
    """分析输出分布"""
    print("\n" + "="*80)
    print("输出分布对比")
    print("="*80)
    
    frets_model.eval()
    t3time_model.eval()
    
    frets_outputs = []
    t3time_outputs = []
    trues = []
    
    with torch.no_grad():
        for bx, by, bxm, bym, emb in data_loader:
            bx = bx.to(device).float()
            emb = emb.to(device).float()
            by_pred = by[:, -96:, :].to(device).float()
            
            frets_out = frets_model(bx, None, emb)
            t3time_out = t3time_model(bx, None, emb)
            
            frets_outputs.append(frets_out.cpu())
            t3time_outputs.append(t3time_out.cpu())
            trues.append(by_pred.cpu())
            
            if len(frets_outputs) >= 10:  # 只分析前10个batch
                break
    
    frets_outputs = torch.cat(frets_outputs, dim=0)
    t3time_outputs = torch.cat(t3time_outputs, dim=0)
    trues = torch.cat(trues, dim=0)
    
    # 计算误差分布
    frets_error = (frets_outputs - trues).abs()
    t3time_error = (t3time_outputs - trues).abs()
    
    print("\n[1] 输出统计:")
    print(f"  FreTS: mean={frets_outputs.mean():.4f}, std={frets_outputs.std():.4f}")
    print(f"  T3Time: mean={t3time_outputs.mean():.4f}, std={t3time_outputs.std():.4f}")
    print(f"  True: mean={trues.mean():.4f}, std={trues.std():.4f}")
    
    print("\n[2] 误差分布:")
    print(f"  FreTS: mean={frets_error.mean():.4f}, std={frets_error.std():.4f}, max={frets_error.max():.4f}")
    print(f"  T3Time: mean={t3time_error.mean():.4f}, std={t3time_error.std():.4f}, max={t3time_error.max():.4f}")
    
    # 计算分位数
    for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
        frets_q = torch.quantile(frets_error, q).item()
        t3time_q = torch.quantile(t3time_error, q).item()
        print(f"  {q*100:.0f}% 分位数: FreTS={frets_q:.4f}, T3Time={t3time_q:.4f}")

def generate_recommendations():
    """生成改进建议"""
    print("\n" + "="*80)
    print("改进建议")
    print("="*80)
    
    recommendations = [
        {
            "优先级": "高",
            "问题": "频域处理方式差异",
            "分析": "T3Time 使用可学习小波包分解（4个频带），而 FreTS 使用单频域表示",
            "建议": [
                "1. 考虑在 FreTS 中引入多频带分解",
                "2. 或者增强 FreTS Component 的容量",
                "3. 尝试调整 sparsity_threshold 参数"
            ]
        },
        {
            "优先级": "高",
            "问题": "正则化强度不足",
            "分析": "FreTS 使用 dropout=0.1, weight_decay=1e-4，而 T3Time 使用 dropout=0.5, weight_decay=1e-3",
            "建议": [
                "1. 尝试增加 dropout 到 0.3-0.5",
                "2. 增加 weight_decay 到 1e-3",
                "3. 观察是否过拟合"
            ]
        },
        {
            "优先级": "中",
            "问题": "模型容量差异",
            "分析": "FreTS 使用 channel=64，而 T3Time 使用 channel=128",
            "建议": [
                "1. 尝试增加 channel 到 128",
                "2. 但要注意配合更强的正则化"
            ]
        },
        {
            "优先级": "中",
            "问题": "融合机制可能不够有效",
            "分析": "Gate 融合虽然引入了 horizon 信息，但可能不如 T3Time 的静态权重+门控组合",
            "建议": [
                "1. 尝试使用 T3Time 的静态权重初始化策略",
                "2. 考虑引入先验引导初始化",
                "3. 尝试不同的门控网络结构"
            ]
        },
        {
            "优先级": "低",
            "问题": "训练策略差异",
            "分析": "可能需要调整学习率调度器或其他训练超参数",
            "建议": [
                "1. 尝试使用 Step Decay（T3Time V30 使用）",
                "2. 调整学习率",
                "3. 尝试不同的优化器"
            ]
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n[{i}] {rec['优先级']}优先级 - {rec['问题']}")
        print(f"    分析: {rec['分析']}")
        print(f"    建议:")
        for suggestion in rec['建议']:
            print(f"      {suggestion}")

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
    
    # 对比架构
    frets_model, t3time_model = compare_architectures()
    
    # 分析特征统计
    analyze_feature_statistics(frets_model, t3time_model, data_loader, device)
    
    # 分析输出分布
    analyze_output_distribution(frets_model, t3time_model, data_loader, device)
    
    # 生成建议
    generate_recommendations()
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
