#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比原版 T3Time 和 T3Time_FreTS_Gated_Qwen_LongSeq_v2 的结果
"""
import json
import os
import sys
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_results(result_file=None):
    if result_file is None:
        result_file = os.path.join(project_root, "experiment_results.log")
    
    results = {
        "T3Time": [],
        "T3Time_FreTS_Gated_Qwen_LongSeq_v2": []
    }
    
    if not os.path.exists(result_file):
        print(f"❌ 结果文件不存在: {result_file}")
        return results
    
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                model = data.get('model', '')
                model_id = data.get('model_id', '')
                
                # 匹配原版 T3Time（model 为 "T3Time" 且没有 LongSeq 相关标识）
                if model == 'T3Time' and 'LongSeq' not in model_id and 'LongSeq' not in model:
                    results['T3Time'].append(data)
                # 匹配 v2 版本
                elif model == 'T3Time_FreTS_Gated_Qwen_LongSeq_v2' or model_id.startswith('T3Time_FreTS_Gated_Qwen_LongSeq_v2'):
                    results['T3Time_FreTS_Gated_Qwen_LongSeq_v2'].append(data)
            except:
                continue
    
    return results

def compare_results(results):
    print("="*80)
    print("原版 T3Time vs T3Time_FreTS_Gated_Qwen_LongSeq_v2 对比分析")
    print("="*80)
    
    # 按预测长度组织数据
    pred_len_data = defaultdict(lambda: defaultdict(list))
    
    for model_name, model_results in results.items():
        for r in model_results:
            pred_len = r.get('pred_len', 'unknown')
            # 将 pred_len 转换为字符串作为 key
            pred_len_str = str(pred_len)
            if pred_len_str.isdigit():
                pred_len_data[pred_len_str][model_name].append({
                    'mse': r.get('test_mse', float('inf')),
                    'mae': r.get('test_mae', float('inf')),
                    'seed': r.get('seed', 'unknown')
                })
    
    if not pred_len_data:
        print("\n❌ 未找到实验结果")
        return
    
    # 打印对比表格
    print("\n" + "="*80)
    print("按预测长度对比（MSE）")
    print("="*80)
    
    pred_lens = sorted([int(k) for k in pred_len_data.keys() if str(k).isdigit()])
    
    print(f"{'预测长度':<12} {'原版T3Time':<20} {'v2版本':<20} {'改进幅度':<15}")
    print("-"*80)
    
    for pred_len in pred_lens:
        pred_len_str = str(pred_len)
        row = f"{pred_len:<12}"
        
        t3time_mse = None
        v2_mse = None
        
        if 'T3Time' in pred_len_data[pred_len_str]:
            stats = pred_len_data[pred_len_str]['T3Time']
            mse_list = [s['mse'] for s in stats]
            t3time_mse = sum(mse_list) / len(mse_list)
            row += f" {t3time_mse:<20.6f}"
        else:
            row += f" {'N/A':<20}"
        
        if 'T3Time_FreTS_Gated_Qwen_LongSeq_v2' in pred_len_data[pred_len_str]:
            stats = pred_len_data[pred_len_str]['T3Time_FreTS_Gated_Qwen_LongSeq_v2']
            mse_list = [s['mse'] for s in stats]
            v2_mse = sum(mse_list) / len(mse_list)
            row += f" {v2_mse:<20.6f}"
        else:
            row += f" {'N/A':<20}"
        
        if t3time_mse and v2_mse:
            improvement = (t3time_mse - v2_mse) / t3time_mse * 100
            row += f" {improvement:+.2f}%"
        else:
            row += f" {'N/A':<15}"
        
        print(row)
    
    # MAE 对比
    print("\n" + "="*80)
    print("按预测长度对比（MAE）")
    print("="*80)
    
    print(f"{'预测长度':<12} {'原版T3Time':<20} {'v2版本':<20} {'改进幅度':<15}")
    print("-"*80)
    
    for pred_len in pred_lens:
        pred_len_str = str(pred_len)
        row = f"{pred_len:<12}"
        
        t3time_mae = None
        v2_mae = None
        
        if 'T3Time' in pred_len_data[pred_len_str]:
            stats = pred_len_data[pred_len_str]['T3Time']
            mae_list = [s['mae'] for s in stats]
            t3time_mae = sum(mae_list) / len(mae_list)
            row += f" {t3time_mae:<20.6f}"
        else:
            row += f" {'N/A':<20}"
        
        if 'T3Time_FreTS_Gated_Qwen_LongSeq_v2' in pred_len_data[pred_len_str]:
            stats = pred_len_data[pred_len_str]['T3Time_FreTS_Gated_Qwen_LongSeq_v2']
            mae_list = [s['mae'] for s in stats]
            v2_mae = sum(mae_list) / len(mae_list)
            row += f" {v2_mae:<20.6f}"
        else:
            row += f" {'N/A':<20}"
        
        if t3time_mae and v2_mae:
            improvement = (t3time_mae - v2_mae) / t3time_mae * 100
            row += f" {improvement:+.2f}%"
        else:
            row += f" {'N/A':<15}"
        
        print(row)
    
    # 详细统计
    print("\n" + "="*80)
    print("详细统计（按预测长度）")
    print("="*80)
    
    for pred_len in pred_lens:
        pred_len_str = str(pred_len)
        print(f"\n预测长度: {pred_len}")
        print("-"*80)
        
        for model_name in ['T3Time', 'T3Time_FreTS_Gated_Qwen_LongSeq_v2']:
            if model_name in pred_len_data[pred_len_str]:
                stats = pred_len_data[pred_len_str][model_name]
                mse_list = [s['mse'] for s in stats]
                mae_list = [s['mae'] for s in stats]
                
                print(f"  {model_name}:")
                print(f"    实验次数: {len(stats)}")
                print(f"    MSE: 平均={sum(mse_list)/len(mse_list):.6f}, 最小={min(mse_list):.6f}, 最大={max(mse_list):.6f}")
                print(f"    MAE: 平均={sum(mae_list)/len(mae_list):.6f}, 最小={min(mae_list):.6f}, 最大={max(mae_list):.6f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='对比原版 T3Time 和 v2 版本的结果')
    parser.add_argument('--result_file', type=str, default=None, help='结果文件路径')
    args = parser.parse_args()
    
    results = load_results(args.result_file)
    compare_results(results)
    
    print("\n" + "="*80)
    print("对比分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
