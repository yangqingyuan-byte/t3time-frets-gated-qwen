#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 T3Time_FreTS_Gated_Qwen_LongSeq 实验结果
对比不同预测长度的性能
"""
import json
import os
import sys
from collections import defaultdict

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_results(result_file=None, model_name="T3Time_FreTS_Gated_Qwen_LongSeq"):
    """加载实验结果"""
    if result_file is None:
        result_file = os.path.join(project_root, "experiment_results.log")
    
    results = []
    
    if not os.path.exists(result_file):
        print(f"❌ 结果文件不存在: {result_file}")
        return results
    
    with open(result_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                if data.get('model') == model_name or data.get('model_id', '').startswith(model_name):
                    results.append(data)
            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue
    
    return results

def analyze_by_pred_len(results):
    """按预测长度分析结果"""
    pred_len_stats = defaultdict(list)
    
    for r in results:
        pred_len = r.get('pred_len', 'unknown')
        pred_len_stats[pred_len].append({
            'mse': r.get('test_mse', float('inf')),
            'mae': r.get('test_mae', float('inf')),
            'seed': r.get('seed', 'unknown'),
            'model_id': r.get('model_id', 'unknown')
        })
    
    return pred_len_stats

def analyze_by_config(results):
    """按配置分析结果"""
    config_stats = defaultdict(list)
    
    for r in results:
        # 构建配置键
        config_key = (
            r.get('horizon_norm', 'unknown'),
            r.get('fusion_mode', 'unknown'),
            r.get('use_dynamic_sparsity', False)
        )
        config_stats[config_key].append({
            'pred_len': r.get('pred_len', 'unknown'),
            'mse': r.get('test_mse', float('inf')),
            'mae': r.get('test_mae', float('inf')),
            'seed': r.get('seed', 'unknown')
        })
    
    return config_stats

def print_analysis(results):
    """打印分析结果"""
    print("="*80)
    print("T3Time_FreTS_Gated_Qwen_LongSeq 实验结果分析")
    print("="*80)
    
    if not results:
        print("\n❌ 未找到实验结果")
        return
    
    print(f"\n找到 {len(results)} 条实验结果\n")
    
    # 按预测长度分析
    pred_len_stats = analyze_by_pred_len(results)
    
    print("="*80)
    print("按预测长度分析")
    print("="*80)
    print(f"{'预测长度':<12} {'实验次数':<10} {'平均MSE':<15} {'最小MSE':<15} {'最大MSE':<15} {'平均MAE':<15}")
    print("-"*80)
    
    for pred_len in sorted(pred_len_stats.keys()):
        stats = pred_len_stats[pred_len]
        mse_list = [s['mse'] for s in stats]
        mae_list = [s['mae'] for s in stats]
        
        print(f"{pred_len:<12} {len(stats):<10} "
              f"{sum(mse_list)/len(mse_list):<15.6f} "
              f"{min(mse_list):<15.6f} "
              f"{max(mse_list):<15.6f} "
              f"{sum(mae_list)/len(mae_list):<15.6f}")
    
    # 按配置分析
    config_stats = analyze_by_config(results)
    
    print("\n" + "="*80)
    print("按配置分析")
    print("="*80)
    print(f"{'Horizon归一化':<15} {'融合模式':<15} {'动态稀疏化':<12} {'实验次数':<10} {'平均MSE':<15} {'平均MAE':<15}")
    print("-"*80)
    
    for (horizon_norm, fusion_mode, use_dynamic) in sorted(config_stats.keys()):
        stats = config_stats[(horizon_norm, fusion_mode, use_dynamic)]
        mse_list = [s['mse'] for s in stats]
        mae_list = [s['mae'] for s in stats]
        
        print(f"{horizon_norm:<15} {fusion_mode:<15} {str(use_dynamic):<12} {len(stats):<10} "
              f"{sum(mse_list)/len(mse_list):<15.6f} "
              f"{sum(mae_list)/len(mae_list):<15.6f}")
    
    # 最佳结果
    print("\n" + "="*80)
    print("最佳结果（按MSE）")
    print("="*80)
    best_mse = min(results, key=lambda x: x.get('test_mse', float('inf')))
    print(f"预测长度: {best_mse.get('pred_len', 'N/A')}")
    print(f"MSE: {best_mse.get('test_mse', 'N/A'):.6f}")
    print(f"MAE: {best_mse.get('test_mae', 'N/A'):.6f}")
    print(f"配置: horizon_norm={best_mse.get('horizon_norm', 'N/A')}, "
          f"fusion_mode={best_mse.get('fusion_mode', 'N/A')}, "
          f"use_dynamic_sparsity={best_mse.get('use_dynamic_sparsity', 'N/A')}")
    print(f"Seed: {best_mse.get('seed', 'N/A')}")
    print(f"Model ID: {best_mse.get('model_id', 'N/A')}")
    
    # 对比不同预测长度的最佳结果
    print("\n" + "="*80)
    print("各预测长度的最佳结果")
    print("="*80)
    print(f"{'预测长度':<12} {'MSE':<15} {'MAE':<15} {'配置':<50}")
    print("-"*80)
    
    for pred_len in sorted(pred_len_stats.keys()):
        stats = pred_len_stats[pred_len]
        best = min(stats, key=lambda x: x['mse'])
        config_str = f"horizon={best.get('horizon_norm', 'N/A')}, fusion={best.get('fusion_mode', 'N/A')}"
        print(f"{pred_len:<12} {best['mse']:<15.6f} {best['mae']:<15.6f} {config_str:<50}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='分析 T3Time_FreTS_Gated_Qwen_LongSeq 实验结果')
    parser.add_argument('--result_file', type=str, default=None, help='结果文件路径（默认: experiment_results.log）')
    parser.add_argument('--model_name', type=str, default='T3Time_FreTS_Gated_Qwen_LongSeq', help='模型名称')
    
    args = parser.parse_args()
    
    results = load_results(args.result_file, args.model_name)
    print_analysis(results)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
