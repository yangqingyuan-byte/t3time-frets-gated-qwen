#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 seed 2000-2100 多种子寻优实验结果
固定配置：Channel=64, Dropout=0.5, Head=8, Pred_Len=720
"""
import json
import os
import sys
from collections import defaultdict

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_seed_search_results(result_file=None, seed_start=2000, seed_end=2100, 
                             pred_len=720, model_id_prefix="T3Time_FreTS_Gated_Qwen_Hyperopt"):
    """加载多种子寻优实验结果"""
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
                # 检查是否符合条件
                seed = data.get('seed')
                pred = data.get('pred_len')
                model_id = data.get('model_id', '')
                
                if (seed_start <= seed <= seed_end and 
                    pred == pred_len and
                    model_id.startswith(model_id_prefix)):
                    results.append(data)
            except json.JSONDecodeError as e:
                continue
            except Exception as e:
                continue
    
    return results

def analyze_seed_results(results):
    """分析多种子寻优结果"""
    if not results:
        print("❌ 未找到符合条件的实验结果")
        return
    
    print("="*80)
    print(f"T3Time_FreTS_Gated_Qwen 多种子寻优结果分析 (Seed 2000-2100, Pred_Len=720)")
    print("="*80)
    print(f"\n找到 {len(results)} 条实验结果\n")
    
    # 按 MSE 排序
    sorted_results_mse = sorted(results, key=lambda x: x.get('test_mse', float('inf')))
    best_mse = sorted_results_mse[0] if sorted_results_mse else None
    
    # 按 MAE 排序
    sorted_results_mae = sorted(results, key=lambda x: x.get('test_mae', float('inf')))
    best_mae = sorted_results_mae[0] if sorted_results_mae else None
    
    # 统计信息
    mse_values = [r.get('test_mse', 0) for r in results]
    mae_values = [r.get('test_mae', 0) for r in results]
    
    mse_mean = sum(mse_values) / len(mse_values) if mse_values else 0
    mse_min = min(mse_values) if mse_values else 0
    mse_max = max(mse_values) if mse_values else 0
    mse_std = (sum((x - mse_mean) ** 2 for x in mse_values) / len(mse_values)) ** 0.5 if mse_values else 0
    
    mae_mean = sum(mae_values) / len(mae_values) if mae_values else 0
    mae_min = min(mae_values) if mae_values else 0
    mae_max = max(mae_values) if mae_values else 0
    mae_std = (sum((x - mae_mean) ** 2 for x in mae_values) / len(mae_values)) ** 0.5 if mae_values else 0
    
    # 显示最佳 MSE 结果
    print("="*80)
    print("🏆 最佳 MSE 结果")
    print("="*80)
    if best_mse:
        print(f"Seed:           {best_mse.get('seed', 'N/A')}")
        print(f"Test MSE:       {best_mse.get('test_mse', 'N/A'):.6f}")
        print(f"Test MAE:       {best_mse.get('test_mae', 'N/A'):.6f}")
        print(f"Channel:        {best_mse.get('channel', 'N/A')}")
        print(f"Dropout:        {best_mse.get('dropout_n', 'N/A')}")
        print(f"Head:           {best_mse.get('head', 'N/A')}")
        print(f"Learning Rate:  {best_mse.get('learning_rate', 'N/A')}")
        print(f"Weight Decay:   {best_mse.get('weight_decay', 'N/A')}")
        print(f"Batch Size:     {best_mse.get('batch_size', 'N/A')}")
        print(f"Loss Function:  {best_mse.get('loss_fn', 'N/A')}")
        print(f"Timestamp:      {best_mse.get('timestamp', 'N/A')}")
    
    # 显示最佳 MAE 结果
    print("\n" + "="*80)
    print("🏆 最佳 MAE 结果")
    print("="*80)
    if best_mae:
        print(f"Seed:           {best_mae.get('seed', 'N/A')}")
        print(f"Test MSE:       {best_mae.get('test_mse', 'N/A'):.6f}")
        print(f"Test MAE:       {best_mae.get('test_mae', 'N/A'):.6f}")
        print(f"Channel:        {best_mae.get('channel', 'N/A')}")
        print(f"Dropout:        {best_mae.get('dropout_n', 'N/A')}")
        print(f"Head:           {best_mae.get('head', 'N/A')}")
        print(f"Learning Rate:  {best_mae.get('learning_rate', 'N/A')}")
        print(f"Weight Decay:   {best_mae.get('weight_decay', 'N/A')}")
        print(f"Batch Size:     {best_mae.get('batch_size', 'N/A')}")
        print(f"Loss Function:  {best_mae.get('loss_fn', 'N/A')}")
        print(f"Timestamp:      {best_mae.get('timestamp', 'N/A')}")
    
    # 检查最佳 MSE 和 MAE 是否来自同一个种子
    if best_mse and best_mae:
        if best_mse.get('seed') == best_mae.get('seed'):
            print("\n✅ 最佳 MSE 和最佳 MAE 来自同一个种子！")
        else:
            print(f"\n⚠️  最佳 MSE (Seed {best_mse.get('seed')}) 和最佳 MAE (Seed {best_mae.get('seed')}) 来自不同种子")
    
    # 统计信息
    print("\n" + "="*80)
    print("📊 统计信息")
    print("="*80)
    print(f"总实验数:        {len(results)}")
    print(f"\nMSE 统计:")
    print(f"  均值:          {mse_mean:.6f}")
    print(f"  最小值:        {mse_min:.6f}")
    print(f"  最大值:        {mse_max:.6f}")
    print(f"  标准差:        {mse_std:.6f}")
    print(f"\nMAE 统计:")
    print(f"  均值:          {mae_mean:.6f}")
    print(f"  最小值:        {mae_min:.6f}")
    print(f"  最大值:        {mae_max:.6f}")
    print(f"  标准差:        {mae_std:.6f}")
    
    # Top 10 最佳结果（按 MSE）
    print("\n" + "="*80)
    print("Top 10 最佳配置（按 MSE 排序）")
    print("="*80)
    print(f"{'Rank':<6} {'Seed':<8} {'MSE':<15} {'MAE':<15}")
    print("-"*80)
    
    for i, r in enumerate(sorted_results_mse[:10], 1):
        print(f"{i:<6} {r.get('seed', 'N/A'):<8} "
              f"{r.get('test_mse', 'N/A'):<15.6f} {r.get('test_mae', 'N/A'):<15.6f}")
    
    # Top 10 最佳结果（按 MAE）
    print("\n" + "="*80)
    print("Top 10 最佳配置（按 MAE 排序）")
    print("="*80)
    print(f"{'Rank':<6} {'Seed':<8} {'MSE':<15} {'MAE':<15}")
    print("-"*80)
    
    for i, r in enumerate(sorted_results_mae[:10], 1):
        print(f"{i:<6} {r.get('seed', 'N/A'):<8} "
              f"{r.get('test_mse', 'N/A'):<15.6f} {r.get('test_mae', 'N/A'):<15.6f}")
    
    # 种子分布分析（按 MSE 区间）
    print("\n" + "="*80)
    print("种子分布分析（按 MSE 区间）")
    print("="*80)
    
    # 定义 MSE 区间
    mse_ranges = [
        (0, mse_mean - mse_std, "优秀 (< 均值-1σ)"),
        (mse_mean - mse_std, mse_mean, "良好 (均值-1σ ~ 均值)"),
        (mse_mean, mse_mean + mse_std, "一般 (均值 ~ 均值+1σ)"),
        (mse_mean + mse_std, float('inf'), "较差 (> 均值+1σ)")
    ]
    
    for min_val, max_val, label in mse_ranges:
        count = sum(1 for mse in mse_values if min_val <= mse < max_val)
        percentage = count / len(mse_values) * 100 if mse_values else 0
        print(f"{label:<25} {count:>4} 个种子 ({percentage:>5.1f}%)")
    
    # 显示最佳 MSE 的命令行格式
    if best_mse:
        print("\n" + "="*80)
        print("📋 最佳 MSE 参数组合（命令行格式）")
        print("="*80)
        print("python train_frets_gated_qwen.py \\")
        print(f"    --data_path {best_mse.get('data_path', 'ETTh1')} \\")
        print(f"    --seq_len {best_mse.get('seq_len', 96)} \\")
        print(f"    --pred_len {best_mse.get('pred_len', 720)} \\")
        print(f"    --channel {best_mse.get('channel', 64)} \\")
        print(f"    --head {best_mse.get('head', 8)} \\")
        print(f"    --e_layer {best_mse.get('e_layer', 1)} \\")
        print(f"    --d_layer {best_mse.get('d_layer', 1)} \\")
        print(f"    --learning_rate {best_mse.get('learning_rate', '1e-4')} \\")
        print(f"    --weight_decay {best_mse.get('weight_decay', '1e-4')} \\")
        print(f"    --dropout_n {best_mse.get('dropout_n', 0.5)} \\")
        print(f"    --batch_size {best_mse.get('batch_size', 16)} \\")
        print(f"    --loss_fn {best_mse.get('loss_fn', 'mse')} \\")
        print(f"    --lradj {best_mse.get('lradj', 'type1')} \\")
        print(f"    --embed_version {best_mse.get('embed_version', 'qwen3_0.6b')} \\")
        print(f"    --epochs {best_mse.get('epochs', 150)} \\")
        print(f"    --es_patience {best_mse.get('patience', 10)} \\")
        print(f"    --seed {best_mse.get('seed', 'N/A')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析多种子寻优实验结果')
    parser.add_argument('--result_file', type=str, default=None, help='实验结果文件路径')
    parser.add_argument('--seed_start', type=int, default=2000, help='种子起始值')
    parser.add_argument('--seed_end', type=int, default=2100, help='种子结束值')
    parser.add_argument('--pred_len', type=int, default=720, help='预测长度')
    parser.add_argument('--model_id', type=str, default="T3Time_FreTS_Gated_Qwen_Hyperopt", 
                       help='模型ID前缀')
    
    args = parser.parse_args()
    
    results = load_seed_search_results(
        result_file=args.result_file,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        pred_len=args.pred_len,
        model_id_prefix=args.model_id
    )
    
    analyze_seed_results(results)
