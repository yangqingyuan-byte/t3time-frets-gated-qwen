#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 T3Time_FreTS_Gated_Qwen_LongSeq_v2 实验结果
"""
import json
import os
import sys
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_results(result_file=None, model_name="T3Time_FreTS_Gated_Qwen_LongSeq_v2"):
    if result_file is None:
        result_file = os.path.join(project_root, "experiment_results.log")
    
    results = []
    
    if not os.path.exists(result_file):
        print(f"❌ 结果文件不存在: {result_file}")
        return results
    
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                if data.get('model') == model_name or data.get('model_id', '').startswith(model_name):
                    results.append(data)
            except:
                continue
    
    return results

def analyze_results(results):
    print("="*80)
    print("T3Time_FreTS_Gated_Qwen_LongSeq_v2 实验结果分析")
    print("="*80)
    
    if not results:
        print("\n❌ 未找到实验结果")
        return
    
    print(f"\n找到 {len(results)} 条实验结果\n")
    
    # 按预测长度分析
    pred_len_stats = defaultdict(list)
    for r in results:
        pred_len = r.get('pred_len', 'unknown')
        pred_len_stats[pred_len].append({
            'mse': r.get('test_mse', float('inf')),
            'mae': r.get('test_mae', float('inf')),
            'seed': r.get('seed', 'unknown'),
            'model_id': r.get('model_id', 'unknown')
        })
    
    print("="*80)
    print("按预测长度分析")
    print("="*80)
    print(f"{'预测长度':<12} {'实验次数':<10} {'平均MSE':<15} {'最小MSE':<15} {'最大MSE':<15} {'平均MAE':<15}")
    print("-"*80)
    
    for pred_len in sorted(pred_len_stats.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
        stats = pred_len_stats[pred_len]
        mse_list = [s['mse'] for s in stats]
        mae_list = [s['mae'] for s in stats]
        
        print(f"{pred_len:<12} {len(stats):<10} "
              f"{sum(mse_list)/len(mse_list):<15.6f} "
              f"{min(mse_list):<15.6f} "
              f"{max(mse_list):<15.6f} "
              f"{sum(mae_list)/len(mae_list):<15.6f}")
    
    # 最佳结果
    print("\n" + "="*80)
    print("最佳结果（按MSE）")
    print("="*80)
    best_mse = min(results, key=lambda x: x.get('test_mse', float('inf')))
    print(f"预测长度: {best_mse.get('pred_len', 'N/A')}")
    print(f"MSE: {best_mse.get('test_mse', 'N/A'):.6f}")
    print(f"MAE: {best_mse.get('test_mae', 'N/A'):.6f}")
    print(f"Seed: {best_mse.get('seed', 'N/A')}")
    print(f"Model ID: {best_mse.get('model_id', 'N/A')}")
    
    # 各预测长度的最佳结果
    print("\n" + "="*80)
    print("各预测长度的最佳结果")
    print("="*80)
    print(f"{'预测长度':<12} {'MSE':<15} {'MAE':<15} {'Seed':<10}")
    print("-"*80)
    
    for pred_len in sorted(pred_len_stats.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
        stats = pred_len_stats[pred_len]
        best = min(stats, key=lambda x: x['mse'])
        print(f"{pred_len:<12} {best['mse']:<15.6f} {best['mae']:<15.6f} {best['seed']:<10}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='分析 T3Time_FreTS_Gated_Qwen_LongSeq_v2 实验结果')
    parser.add_argument('--result_file', type=str, default=None, help='结果文件路径')
    args = parser.parse_args()
    
    results = load_results(args.result_file)
    analyze_results(results)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
