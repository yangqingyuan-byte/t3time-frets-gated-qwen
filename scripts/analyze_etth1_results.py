#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 ETTh1 多预测长度实验结果
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_recent_results(result_file=None, data_path="ETTh1", model_name="T3Time", pred_lens=[96, 192, 336, 720], seed=2024, limit=4):
    """加载最近的实验结果"""
    if result_file is None:
        result_file = os.path.join(project_root, "experiment_results.log")
    
    results = []
    
    if not os.path.exists(result_file):
        print(f"❌ 结果文件不存在: {result_file}")
        return results
    
    with open(result_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 从后往前读取，找到最近的匹配结果
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            data = json.loads(line.strip())
            if (data.get('model') == model_name and 
                data.get('data_path') == data_path and
                data.get('pred_len') in pred_lens and
                data.get('seed') == seed):
                results.append(data)
                if len(results) >= limit:
                    break
        except:
            continue
    
    return results

def analyze_results(results):
    print("="*80)
    print("ETTh1 多预测长度实验结果分析")
    print("="*80)
    
    if not results:
        print("\n❌ 未找到实验结果")
        return
    
    print(f"\n找到 {len(results)} 条实验结果\n")
    
    # 按预测长度组织数据
    pred_len_data = {}
    for r in results:
        pred_len = r.get('pred_len')
        if pred_len not in pred_len_data:
            pred_len_data[pred_len] = r
    
    # 按预测长度排序
    pred_lens = sorted([k for k in pred_len_data.keys()])
    
    # 打印结果表格
    print("="*80)
    print("实验结果汇总")
    print("="*80)
    print(f"{'预测长度':<12} {'MSE':<15} {'MAE':<15} {'Channel':<10} {'Dropout':<10} {'E_Layer':<10} {'D_Layer':<10}")
    print("-"*80)
    
    for pred_len in pred_lens:
        r = pred_len_data[pred_len]
        print(f"{pred_len:<12} "
              f"{r.get('test_mse', 'N/A'):<15.6f} "
              f"{r.get('test_mae', 'N/A'):<15.6f} "
              f"{r.get('channel', 'N/A'):<10} "
              f"{r.get('dropout_n', 'N/A'):<10.2f} "
              f"{r.get('e_layer', 'N/A'):<10} "
              f"{r.get('d_layer', 'N/A'):<10}")
    
    # 详细参数
    print("\n" + "="*80)
    print("详细参数配置")
    print("="*80)
    
    for pred_len in pred_lens:
        r = pred_len_data[pred_len]
        print(f"\n预测长度: {pred_len}")
        print("-"*80)
        print(f"  MSE: {r.get('test_mse', 'N/A'):.6f}")
        print(f"  MAE: {r.get('test_mae', 'N/A'):.6f}")
        print(f"  Seed: {r.get('seed', 'N/A')}")
        print(f"  Channel: {r.get('channel', 'N/A')}")
        print(f"  Batch Size: {r.get('batch_size', 'N/A')}")
        print(f"  Learning Rate: {r.get('learning_rate', 'N/A')}")
        print(f"  Dropout: {r.get('dropout_n', 'N/A')}")
        print(f"  E_Layer: {r.get('e_layer', 'N/A')}")
        print(f"  D_Layer: {r.get('d_layer', 'N/A')}")
        print(f"  Head: {r.get('head', 8)}")
        print(f"  Embed Version: {r.get('embed_version', 'N/A')}")
        print(f"  时间戳: {r.get('timestamp', 'N/A')}")
    
    # 性能趋势
    print("\n" + "="*80)
    print("性能趋势分析")
    print("="*80)
    
    mse_list = [pred_len_data[p].get('test_mse', float('inf')) for p in pred_lens]
    mae_list = [pred_len_data[p].get('test_mae', float('inf')) for p in pred_lens]
    
    print(f"\nMSE 变化:")
    for i, pred_len in enumerate(pred_lens):
        if i > 0:
            change = mse_list[i] - mse_list[i-1]
            change_pct = (change / mse_list[i-1]) * 100 if mse_list[i-1] > 0 else 0
            print(f"  {pred_lens[i-1]} -> {pred_len}: {change:+.6f} ({change_pct:+.2f}%)")
        print(f"  {pred_len}: {mse_list[i]:.6f}")
    
    print(f"\nMAE 变化:")
    for i, pred_len in enumerate(pred_lens):
        if i > 0:
            change = mae_list[i] - mae_list[i-1]
            change_pct = (change / mae_list[i-1]) * 100 if mae_list[i-1] > 0 else 0
            print(f"  {pred_lens[i-1]} -> {pred_len}: {change:+.6f} ({change_pct:+.2f}%)")
        print(f"  {pred_len}: {mae_list[i]:.6f}")
    
    # 最佳和最差结果
    print("\n" + "="*80)
    print("最佳和最差结果")
    print("="*80)
    
    best_mse_idx = min(range(len(mse_list)), key=lambda i: mse_list[i])
    worst_mse_idx = max(range(len(mse_list)), key=lambda i: mse_list[i])
    
    print(f"\n最佳 MSE: pred_len={pred_lens[best_mse_idx]}, MSE={mse_list[best_mse_idx]:.6f}, MAE={mae_list[best_mse_idx]:.6f}")
    print(f"最差 MSE: pred_len={pred_lens[worst_mse_idx]}, MSE={mse_list[worst_mse_idx]:.6f}, MAE={mae_list[worst_mse_idx]:.6f}")
    
    best_mae_idx = min(range(len(mae_list)), key=lambda i: mae_list[i])
    worst_mae_idx = max(range(len(mae_list)), key=lambda i: mae_list[i])
    
    print(f"\n最佳 MAE: pred_len={pred_lens[best_mae_idx]}, MSE={mse_list[best_mae_idx]:.6f}, MAE={mae_list[best_mae_idx]:.6f}")
    print(f"最差 MAE: pred_len={pred_lens[worst_mae_idx]}, MSE={mse_list[worst_mae_idx]:.6f}, MAE={mae_list[worst_mae_idx]:.6f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='分析 ETTh1 多预测长度实验结果')
    parser.add_argument('--result_file', type=str, default=None, help='结果文件路径')
    parser.add_argument('--data_path', type=str, default='ETTh1', help='数据集名称')
    parser.add_argument('--seed', type=int, default=2024, help='随机种子')
    parser.add_argument('--pred_lens', type=int, nargs='+', default=[96, 192, 336, 720], help='预测长度列表')
    args = parser.parse_args()
    
    results = load_recent_results(
        result_file=args.result_file,
        data_path=args.data_path,
        seed=args.seed,
        pred_lens=args.pred_lens,
        limit=len(args.pred_lens)
    )
    
    analyze_results(results)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
