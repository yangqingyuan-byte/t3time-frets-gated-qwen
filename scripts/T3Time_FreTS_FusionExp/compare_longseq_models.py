#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比不同长序列优化模型的结果
对比原版、v1改进版、v2改进版（完全参考T3Time）的性能
"""
import json
import os
import sys
from collections import defaultdict

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_results(result_file=None, model_names=None):
    """加载实验结果"""
    if result_file is None:
        result_file = os.path.join(project_root, "experiment_results.log")
    
    if model_names is None:
        model_names = [
            "T3Time_FreTS_Gated_Qwen",  # 原版
            "T3Time_FreTS_Gated_Qwen_LongSeq",  # v1改进版
            "T3Time_FreTS_Gated_Qwen_LongSeq_v2",  # v2改进版（完全参考T3Time）
            "T3Time"  # T3Time原版（作为参考）
        ]
    
    results = {}
    
    if not os.path.exists(result_file):
        print(f"❌ 结果文件不存在: {result_file}")
        return results
    
    with open(result_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                model = data.get('model', '')
                model_id = data.get('model_id', '')
                
                # 匹配模型
                matched_model = None
                for model_name in model_names:
                    if model == model_name or model_id.startswith(model_name):
                        matched_model = model_name
                        break
                
                if matched_model:
                    if matched_model not in results:
                        results[matched_model] = []
                    results[matched_model].append(data)
            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue
    
    return results

def compare_models(results):
    """对比不同模型的结果"""
    print("="*80)
    print("长序列预测模型对比分析")
    print("="*80)
    
    if not results:
        print("\n❌ 未找到实验结果")
        return
    
    # 按预测长度组织数据
    pred_len_data = defaultdict(lambda: defaultdict(list))
    
    for model_name, model_results in results.items():
        for r in model_results:
            pred_len = r.get('pred_len', 'unknown')
            pred_len_data[pred_len][model_name].append({
                'mse': r.get('test_mse', float('inf')),
                'mae': r.get('test_mae', float('inf')),
                'seed': r.get('seed', 'unknown')
            })
    
    # 打印对比表格
    print("\n" + "="*80)
    print("按预测长度对比（MSE）")
    print("="*80)
    
    pred_lens = sorted([int(k) for k in pred_len_data.keys() if str(k).isdigit()])
    
    # 表头
    models = ["T3Time", "T3Time_FreTS_Gated_Qwen", "T3Time_FreTS_Gated_Qwen_LongSeq", "T3Time_FreTS_Gated_Qwen_LongSeq_v2"]
    header = f"{'预测长度':<12}"
    for model in models:
        if any(model in pred_len_data[str(pl)] for pl in pred_lens):
            header += f" {model[:25]:<25}"
    print(header)
    print("-"*80)
    
    # 数据行
    for pred_len in pred_lens:
        pred_len_str = str(pred_len)
        row = f"{pred_len:<12}"
        for model in models:
            if model in pred_len_data[pred_len_str]:
                stats = pred_len_data[pred_len_str][model]
                mse_list = [s['mse'] for s in stats]
                avg_mse = sum(mse_list) / len(mse_list)
                row += f" {avg_mse:<25.6f}"
            else:
                row += f" {'N/A':<25}"
        print(row)
    
    # 打印对比表格（MAE）
    print("\n" + "="*80)
    print("按预测长度对比（MAE）")
    print("="*80)
    
    header = f"{'预测长度':<12}"
    for model in models:
        if any(model in pred_len_data[str(pl)] for pl in pred_lens):
            header += f" {model[:25]:<25}"
    print(header)
    print("-"*80)
    
    for pred_len in pred_lens:
        pred_len_str = str(pred_len)
        row = f"{pred_len:<12}"
        for model in models:
            if model in pred_len_data[pred_len_str]:
                stats = pred_len_data[pred_len_str][model]
                mae_list = [s['mae'] for s in stats]
                avg_mae = sum(mae_list) / len(mae_list)
                row += f" {avg_mae:<25.6f}"
            else:
                row += f" {'N/A':<25}"
        print(row)
    
    # 改进幅度分析
    print("\n" + "="*80)
    print("改进幅度分析（相对于原版 T3Time_FreTS_Gated_Qwen）")
    print("="*80)
    
    baseline = "T3Time_FreTS_Gated_Qwen"
    improved_models = ["T3Time_FreTS_Gated_Qwen_LongSeq", "T3Time_FreTS_Gated_Qwen_LongSeq_v2"]
    
    for pred_len in pred_lens:
        pred_len_str = str(pred_len)
        if baseline not in pred_len_data[pred_len_str]:
            continue
        
        baseline_stats = pred_len_data[pred_len_str][baseline]
        baseline_mse = sum(s['mse'] for s in baseline_stats) / len(baseline_stats)
        
        print(f"\n预测长度: {pred_len}")
        print(f"  原版 MSE: {baseline_mse:.6f}")
        
        for model in improved_models:
            if model in pred_len_data[pred_len_str]:
                stats = pred_len_data[pred_len_str][model]
                mse_list = [s['mse'] for s in stats]
                avg_mse = sum(mse_list) / len(mse_list)
                improvement = (baseline_mse - avg_mse) / baseline_mse * 100
                print(f"  {model}: MSE={avg_mse:.6f}, 改进={improvement:+.2f}%")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='对比不同长序列优化模型的结果')
    parser.add_argument('--result_file', type=str, default=None, help='结果文件路径（默认: experiment_results.log）')
    
    args = parser.parse_args()
    
    results = load_results(args.result_file)
    compare_models(results)
    
    print("\n" + "="*80)
    print("对比分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
