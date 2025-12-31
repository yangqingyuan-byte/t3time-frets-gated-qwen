#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验结果分析脚本
自动扫描日志文件，提取实验结果，生成CSV报告

使用方法:
    python scripts/分析实验结果.py [--dataset DATASET_NAME]

功能:
    1. 扫描指定目录下的所有.log文件
    2. 提取MSE和MAE值
    3. 识别任务类型（如i96_o96）
    4. 生成详细结果汇总和最佳结果汇总
"""

import os
import re
import csv
from pathlib import Path
from collections import defaultdict
import argparse


def parse_log_file(log_path):
    """
    解析单个log文件，提取MSE和MAE
    
    Args:
        log_path: 日志文件路径
        
    Returns:
        dict: {
            'mse': float or None,
            'mae': float or None,
            'status': 'success' or 'failed',
            'task_type': str (e.g., 'i96_o96')
        }
    """
    result = {
        'mse': None,
        'mae': None,
        'status': 'failed',
        'task_type': None
    }
    
    try:
        # 从文件名提取任务类型
        log_name = os.path.basename(log_path)
        match = re.search(r'(i\d+_o\d+)', log_name)
        if match:
            result['task_type'] = match.group(1)
        
        # 检查文件是否存在
        if not os.path.exists(log_path):
            result['status'] = 'error: file not found'
            return result
        
        # 从日志内容提取MSE和MAE
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # 检查是否训练完成
        has_training_ends = any('Training ends' in line for line in lines)
        
        # 从后往前查找（结果通常在最后）
        for line in reversed(lines):
            # 匹配格式: On average horizons, Test MSE: 0.4660, Test MAE: 0.4521
            match = re.search(
                r'On average horizons, Test MSE: ([\d.]+), Test MAE: ([\d.]+)',
                line
            )
            if match:
                result['mse'] = float(match.group(1))
                result['mae'] = float(match.group(2))
                result['status'] = 'success'
                break
        
        # 如果没有找到结果，但训练已完成，标记为completed_but_no_test
        if result['status'] == 'failed' and has_training_ends:
            result['status'] = 'completed_but_no_test'
            
    except Exception as e:
        result['status'] = f'error: {str(e)}'
    
    return result


def find_best_results(results):
    """
    找出每个任务类型的最佳结果
    
    Args:
        results: dict, {experiment_name: result_dict}
        
    Returns:
        dict: {
            task_type: {
                'combined': best_experiment,
                'best_mse': mse_best_experiment,
                'best_mae': mae_best_experiment
            }
        }
    """
    best_results = {}
    
    # 按任务类型分组
    task_groups = defaultdict(list)
    for exp_name, result in results.items():
        task_type = result['task_type']
        if task_type and result['status'] == 'success':
            task_groups[task_type].append({
                'experiment': exp_name,
                'mse': result['mse'],
                'mae': result['mae']
            })
    
    # 找出最佳结果
    for task_type, experiments in task_groups.items():
        if not experiments:
            continue
            
        # 综合最佳：MSE和MAE的加权和最小
        combined_best = min(
            experiments,
            key=lambda x: (x['mse'] + x['mae'], x['mse'])
        )
        
        # MSE最佳
        mse_best = min(experiments, key=lambda x: x['mse'])
        
        # MAE最佳
        mae_best = min(experiments, key=lambda x: x['mae'])
        
        best_results[task_type] = {
            'combined': combined_best,
            'best_mse': mse_best,
            'best_mae': mae_best,
        }
    
    return best_results


def analyze_results(results_dir, dataset_name):
    """
    分析实验结果
    
    Args:
        results_dir: 结果目录路径
        dataset_name: 数据集名称
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"错误: 目录不存在: {results_dir}")
        return
    
    # 扫描所有.log文件
    log_files = list(results_dir.glob('*.log'))
    
    if not log_files:
        print(f"警告: 在 {results_dir} 中未找到.log文件")
        return
    
    print(f"找到 {len(log_files)} 个日志文件")
    
    # 解析所有日志文件
    results = {}
    for log_file in log_files:
        log_name = log_file.name
        exp_name = f"{dataset_name}_{log_name.replace('.log', '')}"
        
        print(f"解析: {log_name}")
        result = parse_log_file(log_file)
        results[exp_name] = result
    
    # 统计
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    print(f"\n解析完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    
    # 生成详细结果汇总
    output_csv = results_dir / 'results_summary.csv'
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['实验名', '任务类型', 'MSE', 'MAE', '状态'])
        
        for exp_name, result in sorted(results.items()):
            writer.writerow([
                exp_name,
                result['task_type'] or '',
                f"{result['mse']:.6f}" if result['mse'] is not None else '',
                f"{result['mae']:.6f}" if result['mae'] is not None else '',
                result['status']
            ])
    
    print(f"\n详细结果已保存到: {output_csv}")
    
    # 生成最佳结果汇总
    best_results = find_best_results(results)
    
    if best_results:
        best_results_csv = results_dir / 'best_results.csv'
        with open(best_results_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['任务类型', '最佳实验', 'MSE', 'MAE', '备注'])
            
            for task_type, bests in sorted(best_results.items()):
                combined = bests['combined']
                writer.writerow([
                    task_type,
                    combined['experiment'],
                    f"{combined['mse']:.6f}",
                    f"{combined['mae']:.6f}",
                    '综合最佳(MSE+MAE最小)'
                ])
        
        print(f"最佳结果已保存到: {best_results_csv}")
        
        # 打印最佳结果摘要
        print("\n最佳结果摘要:")
        print("-" * 80)
        for task_type, bests in sorted(best_results.items()):
            combined = bests['combined']
            print(f"{task_type:15s} | MSE: {combined['mse']:.6f} | MAE: {combined['mae']:.6f} | {combined['experiment']}")
    else:
        print("\n警告: 未找到成功的结果")


def main():
    parser = argparse.ArgumentParser(description='分析实验结果')
    parser.add_argument(
        '--dataset',
        type=str,
        default='ETTh1',
        help='数据集名称 (默认: ETTh1)'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='结果目录路径 (默认: ./Results/{dataset})'
    )
    
    args = parser.parse_args()
    
    # 确定结果目录
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(f'./Results/{args.dataset}')
    
    print(f"数据集: {args.dataset}")
    print(f"结果目录: {results_dir}")
    print("=" * 80)
    
    analyze_results(results_dir, args.dataset)


if __name__ == '__main__':
    main()

