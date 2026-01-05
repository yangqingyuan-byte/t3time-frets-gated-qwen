#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3Time_FreTS_FusionExp 消融实验结果分析脚本
分析消融实验的结果，生成详细的对比报告
"""
import sys
import os
import json
import pandas as pd
from collections import defaultdict
import numpy as np

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_ablation_results(result_file=None):
    """加载消融实验结果"""
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
                # 检查是否是消融实验结果
                model_id = data.get('model_id', '')
                ablation_exp = data.get('ablation_exp', '')
                
                # 匹配消融实验：model_id 包含 "Ablation" 或者有 ablation_exp 字段
                if 'Ablation' in model_id or ablation_exp:
                    results.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️  第 {line_num} 行 JSON 解析失败: {e}")
                continue
            except Exception as e:
                print(f"⚠️  第 {line_num} 行处理失败: {e}")
                continue
    
    return results

def get_best_result(results, key='test_mse', minimize=True):
    """获取最佳结果（按 MSE 或 MAE）"""
    if not results:
        return None
    if minimize:
        return min(results, key=lambda x: x.get(key, float('inf')))
    else:
        return max(results, key=lambda x: x.get(key, float('-inf')))

def get_statistics(results, key='test_mse'):
    """计算统计信息（均值、标准差、最小值、最大值）"""
    if not results:
        return None
    
    values = [r.get(key, 0) for r in results if key in r]
    if not values:
        return None
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'count': len(values)
    }

def analyze_experiment_1(results):
    """实验1: FreTS Component 的有效性"""
    print("="*80)
    print("实验1: FreTS Component 的有效性")
    print("="*80)
    
    # 1.1 固定 FFT vs FreTS Component
    print("\n[1.1] 固定 FFT vs FreTS Component:")
    print("-" * 80)
    fft_results = [r for r in results if 'A1_FFT_Magnitude' in r.get('ablation_exp', '')]
    frets_results = [r for r in results if 'A1_FreTS_Component' in r.get('ablation_exp', '')]
    
    if fft_results:
        fft_best = get_best_result(fft_results)
        fft_stats = get_statistics(fft_results)
        print(f"  固定 FFT (Magnitude):")
        print(f"    最佳 MSE: {fft_best['test_mse']:.6f}, MAE: {fft_best['test_mae']:.6f}")
        if fft_stats:
            print(f"    均值: {fft_stats['mean']:.6f} ± {fft_stats['std']:.6f} (n={fft_stats['count']})")
    
    if frets_results:
        frets_best = get_best_result(frets_results)
        frets_stats = get_statistics(frets_results)
        print(f"  FreTS Component:")
        print(f"    最佳 MSE: {frets_best['test_mse']:.6f}, MAE: {frets_best['test_mae']:.6f}")
        if frets_stats:
            print(f"    均值: {frets_stats['mean']:.6f} ± {frets_stats['std']:.6f} (n={frets_stats['count']})")
    
    if fft_results and frets_results:
        fft_best = get_best_result(fft_results)
        frets_best = get_best_result(frets_results)
        improvement = fft_best['test_mse'] - frets_best['test_mse']
        improvement_pct = (improvement / fft_best['test_mse']) * 100
        print(f"\n  对比:")
        print(f"    FreTS 相对 FFT 改进: {improvement:.6f} ({improvement_pct:+.2f}%)")
        if improvement < 0:
            print(f"    ⚠️  FreTS 性能不如 FFT")
        else:
            print(f"    ✅ FreTS 性能优于 FFT")
    
    # 1.2 仅幅度 vs 复数信息
    print("\n[1.2] 仅幅度 vs 复数信息 (固定FFT):")
    print("-" * 80)
    magnitude_results = [r for r in results if 'A2_FFT_Magnitude_Only' in r.get('ablation_exp', '')]
    complex_results = [r for r in results if 'A2_FFT_Complex' in r.get('ablation_exp', '')]
    
    if magnitude_results:
        mag_best = get_best_result(magnitude_results)
        mag_stats = get_statistics(magnitude_results)
        print(f"  仅幅度:")
        print(f"    最佳 MSE: {mag_best['test_mse']:.6f}, MAE: {mag_best['test_mae']:.6f}")
        if mag_stats:
            print(f"    均值: {mag_stats['mean']:.6f} ± {mag_stats['std']:.6f} (n={mag_stats['count']})")
    
    if complex_results:
        comp_best = get_best_result(complex_results)
        comp_stats = get_statistics(complex_results)
        print(f"  复数信息:")
        print(f"    最佳 MSE: {comp_best['test_mse']:.6f}, MAE: {comp_best['test_mae']:.6f}")
        if comp_stats:
            print(f"    均值: {comp_stats['mean']:.6f} ± {comp_stats['std']:.6f} (n={comp_stats['count']})")
    
    if magnitude_results and complex_results:
        mag_best = get_best_result(magnitude_results)
        comp_best = get_best_result(complex_results)
        improvement = mag_best['test_mse'] - comp_best['test_mse']
        improvement_pct = (improvement / mag_best['test_mse']) * 100
        print(f"\n  对比:")
        print(f"    复数相对幅度改进: {improvement:.6f} ({improvement_pct:+.2f}%)")
    
    # 1.3 有无稀疏化机制
    print("\n[1.3] 有无稀疏化机制 (FreTS):")
    print("-" * 80)
    no_sparsity_results = [r for r in results if 'A3_FreTS_NoSparsity' in r.get('ablation_exp', '')]
    with_sparsity_results = [r for r in results if 'A3_FreTS_WithSparsity' in r.get('ablation_exp', '')]
    
    if no_sparsity_results:
        no_sparse_best = get_best_result(no_sparsity_results)
        no_sparse_stats = get_statistics(no_sparsity_results)
        print(f"  无稀疏化:")
        print(f"    最佳 MSE: {no_sparse_best['test_mse']:.6f}, MAE: {no_sparse_best['test_mae']:.6f}")
        if no_sparse_stats:
            print(f"    均值: {no_sparse_stats['mean']:.6f} ± {no_sparse_stats['std']:.6f} (n={no_sparse_stats['count']})")
    
    if with_sparsity_results:
        with_sparse_best = get_best_result(with_sparsity_results)
        with_sparse_stats = get_statistics(with_sparsity_results)
        print(f"  有稀疏化:")
        print(f"    最佳 MSE: {with_sparse_best['test_mse']:.6f}, MAE: {with_sparse_best['test_mae']:.6f}")
        if with_sparse_stats:
            print(f"    均值: {with_sparse_stats['mean']:.6f} ± {with_sparse_stats['std']:.6f} (n={with_sparse_stats['count']})")
    
    if no_sparsity_results and with_sparsity_results:
        no_sparse_best = get_best_result(no_sparsity_results)
        with_sparse_best = get_best_result(with_sparsity_results)
        improvement = no_sparse_best['test_mse'] - with_sparse_best['test_mse']
        improvement_pct = (improvement / no_sparse_best['test_mse']) * 100
        print(f"\n  对比:")
        print(f"    稀疏化机制改进: {improvement:.6f} ({improvement_pct:+.2f}%)")

def analyze_experiment_2(results):
    """实验2: 融合机制对比"""
    print("\n" + "="*80)
    print("实验2: 融合机制对比")
    print("="*80)
    
    fusion_results = defaultdict(list)
    for r in results:
        exp = r.get('ablation_exp', '')
        if 'A4_Fusion' in exp:
            fusion_mode = r.get('fusion_mode', 'unknown')
            fusion_results[fusion_mode].append(r)
    
    if not fusion_results:
        print("  未找到融合机制实验结果")
        return
    
    print(f"\n{'融合模式':<20} {'最佳 MSE':<15} {'最佳 MAE':<15} {'均值 MSE':<15} {'标准差':<10}")
    print("-" * 80)
    
    fusion_stats_list = []
    for mode, res_list in sorted(fusion_results.items()):
        best = get_best_result(res_list)
        stats = get_statistics(res_list)
        if stats:
            fusion_stats_list.append((
                mode, 
                best['test_mse'], 
                best['test_mae'],
                stats['mean'],
                stats['std'],
                stats['count']
            ))
            print(f"{mode:<20} {best['test_mse']:<15.6f} {best['test_mae']:<15.6f} {stats['mean']:<15.6f} {stats['std']:<10.6f}")
    
    if fusion_stats_list:
        best_fusion = min(fusion_stats_list, key=lambda x: x[1])
        print(f"\n✅ 最佳融合模式: {best_fusion[0]} (MSE={best_fusion[1]:.6f}, MAE={best_fusion[2]:.6f})")

def analyze_experiment_3(results):
    """实验3: 超参数敏感性分析"""
    print("\n" + "="*80)
    print("实验3: 超参数敏感性分析")
    print("="*80)
    
    # 3.1 scale 参数
    print("\n[3.1] Scale 参数敏感性:")
    print("-" * 80)
    print(f"{'Scale':<12} {'最佳 MSE':<15} {'最佳 MAE':<15} {'均值 MSE':<15} {'标准差':<10}")
    print("-" * 80)
    
    scale_results = defaultdict(list)
    for r in results:
        exp = r.get('ablation_exp', '')
        if 'A5_Scale' in exp:
            scale = r.get('frets_scale', 0)
            scale_results[scale].append(r)
    
    scale_stats_list = []
    for scale in sorted(scale_results.keys()):
        res_list = scale_results[scale]
        best = get_best_result(res_list)
        stats = get_statistics(res_list)
        if stats:
            scale_stats_list.append((scale, best['test_mse'], best['test_mae'], stats['mean'], stats['std']))
            print(f"{scale:<12.3f} {best['test_mse']:<15.6f} {best['test_mae']:<15.6f} {stats['mean']:<15.6f} {stats['std']:<10.6f}")
    
    if scale_stats_list:
        best_scale = min(scale_stats_list, key=lambda x: x[1])
        print(f"\n✅ 最佳 Scale: {best_scale[0]:.3f} (MSE={best_scale[1]:.6f}, MAE={best_scale[2]:.6f})")
    
    # 3.2 sparsity_threshold 参数
    print("\n[3.2] Sparsity Threshold 参数敏感性:")
    print("-" * 80)
    print(f"{'Sparsity':<12} {'最佳 MSE':<15} {'最佳 MAE':<15} {'均值 MSE':<15} {'标准差':<10}")
    print("-" * 80)
    
    sparsity_results = defaultdict(list)
    for r in results:
        exp = r.get('ablation_exp', '')
        if 'A6_Sparsity' in exp:
            sparsity = r.get('sparsity_threshold', 0)
            sparsity_results[sparsity].append(r)
    
    sparsity_stats_list = []
    for sparsity in sorted(sparsity_results.keys()):
        res_list = sparsity_results[sparsity]
        best = get_best_result(res_list)
        stats = get_statistics(res_list)
        if stats:
            sparsity_stats_list.append((sparsity, best['test_mse'], best['test_mae'], stats['mean'], stats['std']))
            print(f"{sparsity:<12.3f} {best['test_mse']:<15.6f} {best['test_mae']:<15.6f} {stats['mean']:<15.6f} {stats['std']:<10.6f}")
    
    if sparsity_stats_list:
        best_sparsity = min(sparsity_stats_list, key=lambda x: x[1])
        print(f"\n✅ 最佳 Sparsity: {best_sparsity[0]:.3f} (MSE={best_sparsity[1]:.6f}, MAE={best_sparsity[2]:.6f})")

def analyze_experiment_4(results):
    """实验4: 门控机制改进的影响"""
    print("\n" + "="*80)
    print("实验4: 门控机制改进的影响")
    print("="*80)
    
    original_results = [r for r in results if 'A7_Original_Gate' in r.get('ablation_exp', '')]
    improved_results = [r for r in results if 'A7_Improved_Gate' in r.get('ablation_exp', '')]
    
    if original_results:
        orig_best = get_best_result(original_results)
        orig_stats = get_statistics(original_results)
        print(f"\n  原始门控:")
        print(f"    最佳 MSE: {orig_best['test_mse']:.6f}, MAE: {orig_best['test_mae']:.6f}")
        if orig_stats:
            print(f"    均值: {orig_stats['mean']:.6f} ± {orig_stats['std']:.6f} (n={orig_stats['count']})")
    
    if improved_results:
        impr_best = get_best_result(improved_results)
        impr_stats = get_statistics(improved_results)
        print(f"  改进门控:")
        print(f"    最佳 MSE: {impr_best['test_mse']:.6f}, MAE: {impr_best['test_mae']:.6f}")
        if impr_stats:
            print(f"    均值: {impr_stats['mean']:.6f} ± {impr_stats['std']:.6f} (n={impr_stats['count']})")
    
    if original_results and improved_results:
        orig_best = get_best_result(original_results)
        impr_best = get_best_result(improved_results)
        improvement = orig_best['test_mse'] - impr_best['test_mse']
        improvement_pct = (improvement / orig_best['test_mse']) * 100
        print(f"\n  对比:")
        print(f"    改进门控相对原始门控改进: {improvement:.6f} ({improvement_pct:+.2f}%)")

def generate_summary_table(results):
    """生成汇总表格"""
    print("\n" + "="*80)
    print("消融实验汇总表")
    print("="*80)
    
    # 按实验分组
    exp_groups = defaultdict(list)
    for r in results:
        exp = r.get('ablation_exp', '')
        if exp:
            # 提取实验组名（A1, A2, A3等）
            parts = exp.split('_')
            if parts and parts[0].startswith('A'):
                group = parts[0]
                exp_groups[group].append(r)
    
    if not exp_groups:
        print("  未找到消融实验结果")
        return
    
    print(f"\n{'实验组':<10} {'实验名称':<35} {'最佳 MSE':<15} {'最佳 MAE':<15} {'实验次数':<10}")
    print("-" * 100)
    
    all_exps = []
    for group in sorted(exp_groups.keys()):
        group_results = exp_groups[group]
        # 按实验名称分组
        exp_names = defaultdict(list)
        for r in group_results:
            exp_name = r.get('ablation_exp', 'unknown')
            exp_names[exp_name].append(r)
        
        for exp_name in sorted(exp_names.keys()):
            res_list = exp_names[exp_name]
            best = get_best_result(res_list)
            stats = get_statistics(res_list)
            count = stats['count'] if stats else len(res_list)
            all_exps.append((group, exp_name, best['test_mse'], best['test_mae'], count))
            print(f"{group:<10} {exp_name:<35} {best['test_mse']:<15.6f} {best['test_mae']:<15.6f} {count:<10}")
    
    return all_exps

def export_to_csv(results, output_file=None):
    """导出结果到 CSV"""
    if output_file is None:
        output_file = os.path.join(project_root, "scripts", "T3Time_FreTS_FusionExp", "ablation_results.csv")
    
    if not results:
        print("  没有结果可导出")
        return
    
    # 准备数据
    data = []
    for r in results:
        row = {
            'ablation_exp': r.get('ablation_exp', ''),
            'fusion_mode': r.get('fusion_mode', ''),
            'use_frets': r.get('use_frets', ''),
            'use_complex': r.get('use_complex', ''),
            'use_sparsity': r.get('use_sparsity', ''),
            'use_improved_gate': r.get('use_improved_gate', ''),
            'frets_scale': r.get('frets_scale', ''),
            'sparsity_threshold': r.get('sparsity_threshold', ''),
            'test_mse': r.get('test_mse', ''),
            'test_mae': r.get('test_mae', ''),
            'seed': r.get('seed', ''),
            'timestamp': r.get('timestamp', '')
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已导出到: {output_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析消融实验结果')
    parser.add_argument('--result_file', type=str, default=None, help='结果文件路径（默认: experiment_results.log）')
    parser.add_argument('--export_csv', action='store_true', help='导出结果到 CSV')
    parser.add_argument('--csv_file', type=str, default=None, help='CSV 输出文件路径')
    
    args = parser.parse_args()
    
    print("="*80)
    print("T3Time_FreTS_FusionExp 消融实验结果分析")
    print("="*80)
    
    results = load_ablation_results(args.result_file)
    
    if not results:
        print("\n❌ 未找到消融实验结果")
        print("请先运行消融实验脚本: bash scripts/T3Time_FreTS_FusionExp/ablation_study.sh")
        return
    
    print(f"\n找到 {len(results)} 条消融实验结果\n")
    
    # 分析各个实验
    analyze_experiment_1(results)
    analyze_experiment_2(results)
    analyze_experiment_3(results)
    analyze_experiment_4(results)
    all_exps = generate_summary_table(results)
    
    # 导出 CSV
    if args.export_csv:
        export_to_csv(results, args.csv_file)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
