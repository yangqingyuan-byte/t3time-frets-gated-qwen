#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析编码器和解码器层数寻优实验结果
重点分析层数对MSE指标的影响
"""
import json
import os
import sys
from collections import defaultdict
import numpy as np

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_layer_search_results(result_file=None, seed=2088, model_id_prefix="T3Time_FreTS_Gated_Qwen_LayerSearch"):
    """加载层数寻优实验结果"""
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
                # 检查是否是层数寻优实验结果
                if (data.get('seed') == seed and 
                    data.get('model_id', '').startswith(model_id_prefix)):
                    results.append(data)
            except json.JSONDecodeError as e:
                continue
            except Exception as e:
                continue
    
    return results

def analyze_layer_impact(results):
    """分析层数对MSE的影响"""
    if not results:
        return None, None, None, None
    
    # 按MSE排序
    sorted_results = sorted(results, key=lambda x: x.get('test_mse', float('inf')))
    
    # 按E_Layer分组统计
    e_layer_stats = defaultdict(list)
    for r in results:
        e_layer = r.get('e_layer')
        mse = r.get('test_mse', float('inf'))
        mae = r.get('test_mae', float('inf'))
        if e_layer is not None:
            e_layer_stats[e_layer].append({'mse': mse, 'mae': mae})
    
    # 按D_Layer分组统计
    d_layer_stats = defaultdict(list)
    for r in results:
        d_layer = r.get('d_layer')
        mse = r.get('test_mse', float('inf'))
        mae = r.get('test_mae', float('inf'))
        if d_layer is not None:
            d_layer_stats[d_layer].append({'mse': mse, 'mae': mae})
    
    # 按(E_Layer, D_Layer)组合统计
    layer_combo_stats = defaultdict(list)
    for r in results:
        e_layer = r.get('e_layer')
        d_layer = r.get('d_layer')
        mse = r.get('test_mse', float('inf'))
        mae = r.get('test_mae', float('inf'))
        if e_layer is not None and d_layer is not None:
            layer_combo_stats[(e_layer, d_layer)].append({'mse': mse, 'mae': mae})
    
    return sorted_results, e_layer_stats, d_layer_stats, layer_combo_stats

def print_analysis(sorted_results, e_layer_stats, d_layer_stats, layer_combo_stats):
    """打印分析结果"""
    print("="*80)
    print("编码器和解码器层数寻优结果分析")
    print("="*80)
    
    if not sorted_results:
        print("\n❌ 未找到实验结果")
        return
    
    total_results = len(sorted_results)
    print(f"\n找到 {total_results} 条实验结果\n")
    
    # 最佳结果
    best = sorted_results[0]
    print("="*80)
    print("🏆 最佳结果（最小MSE）")
    print("="*80)
    print(f"E_Layer:     {best.get('e_layer', 'N/A')}")
    print(f"D_Layer:     {best.get('d_layer', 'N/A')}")
    print(f"MSE:         {best.get('test_mse', 'N/A'):.6f}")
    print(f"MAE:         {best.get('test_mae', 'N/A'):.6f}")
    print(f"Timestamp:   {best.get('timestamp', 'N/A')}")
    
    # 所有结果表格
    print("\n" + "="*80)
    print("所有实验结果（按MSE排序）")
    print("="*80)
    print(f"{'Rank':<6} {'E_Layer':<10} {'D_Layer':<10} {'MSE':<15} {'MAE':<15}")
    print("-"*80)
    
    for i, r in enumerate(sorted_results, 1):
        print(f"{i:<6} {r.get('e_layer', 'N/A'):<10} {r.get('d_layer', 'N/A'):<10} "
              f"{r.get('test_mse', 'N/A'):<15.6f} {r.get('test_mae', 'N/A'):<15.6f}")
    
    # E_Layer 影响分析
    print("\n" + "="*80)
    print("📊 编码器层数 (E_Layer) 对MSE的影响")
    print("="*80)
    print(f"{'E_Layer':<10} {'平均MSE':<15} {'最小MSE':<15} {'最大MSE':<15} {'平均MAE':<15} {'实验数':<8}")
    print("-"*80)
    
    for e_layer in sorted(e_layer_stats.keys()):
        stats = e_layer_stats[e_layer]
        mse_list = [s['mse'] for s in stats]
        mae_list = [s['mae'] for s in stats]
        print(f"{e_layer:<10} "
              f"{np.mean(mse_list):<15.6f} "
              f"{np.min(mse_list):<15.6f} "
              f"{np.max(mse_list):<15.6f} "
              f"{np.mean(mae_list):<15.6f} "
              f"{len(stats):<8}")
    
    # D_Layer 影响分析
    print("\n" + "="*80)
    print("📊 解码器层数 (D_Layer) 对MSE的影响")
    print("="*80)
    print(f"{'D_Layer':<10} {'平均MSE':<15} {'最小MSE':<15} {'最大MSE':<15} {'平均MAE':<15} {'实验数':<8}")
    print("-"*80)
    
    for d_layer in sorted(d_layer_stats.keys()):
        stats = d_layer_stats[d_layer]
        mse_list = [s['mse'] for s in stats]
        mae_list = [s['mae'] for s in stats]
        print(f"{d_layer:<10} "
              f"{np.mean(mse_list):<15.6f} "
              f"{np.min(mse_list):<15.6f} "
              f"{np.max(mse_list):<15.6f} "
              f"{np.mean(mae_list):<15.6f} "
              f"{len(stats):<8}")
    
    # 层数组合影响分析
    print("\n" + "="*80)
    print("📊 层数组合 (E_Layer, D_Layer) 对MSE的影响")
    print("="*80)
    print(f"{'E_Layer':<10} {'D_Layer':<10} {'平均MSE':<15} {'最小MSE':<15} {'最大MSE':<15} {'平均MAE':<15}")
    print("-"*80)
    
    sorted_combos = sorted(layer_combo_stats.items(), key=lambda x: np.mean([s['mse'] for s in x[1]]))
    for (e_layer, d_layer), stats in sorted_combos:
        mse_list = [s['mse'] for s in stats]
        mae_list = [s['mae'] for s in stats]
        print(f"{e_layer:<10} {d_layer:<10} "
              f"{np.mean(mse_list):<15.6f} "
              f"{np.min(mse_list):<15.6f} "
              f"{np.max(mse_list):<15.6f} "
              f"{np.mean(mae_list):<15.6f}")
    
    # 结论分析
    print("\n" + "="*80)
    print("📈 结论分析")
    print("="*80)
    
    # 找出最佳E_Layer
    best_e_layer = min(e_layer_stats.items(), key=lambda x: np.mean([s['mse'] for s in x[1]]))
    print(f"\n最佳编码器层数 (E_Layer): {best_e_layer[0]}")
    print(f"  平均MSE: {np.mean([s['mse'] for s in best_e_layer[1]]):.6f}")
    
    # 找出最佳D_Layer
    best_d_layer = min(d_layer_stats.items(), key=lambda x: np.mean([s['mse'] for s in x[1]]))
    print(f"\n最佳解码器层数 (D_Layer): {best_d_layer[0]}")
    print(f"  平均MSE: {np.mean([s['mse'] for s in best_d_layer[1]]):.6f}")
    
    # 找出最佳组合
    best_combo = min(layer_combo_stats.items(), key=lambda x: np.mean([s['mse'] for s in x[1]]))
    (best_e, best_d), best_combo_stats = best_combo
    print(f"\n最佳层数组合 (E_Layer={best_e}, D_Layer={best_d}):")
    print(f"  平均MSE: {np.mean([s['mse'] for s in best_combo_stats]):.6f}")
    print(f"  最小MSE: {np.min([s['mse'] for s in best_combo_stats]):.6f}")
    
    # 层数影响评估
    print("\n" + "="*80)
    print("🔍 层数影响评估")
    print("="*80)
    
    # 计算E_Layer的MSE变化
    e_layer_mses = {k: np.mean([s['mse'] for s in v]) for k, v in e_layer_stats.items()}
    if len(e_layer_mses) > 1:
        min_e_mse = min(e_layer_mses.values())
        max_e_mse = max(e_layer_mses.values())
        e_impact = ((max_e_mse - min_e_mse) / min_e_mse) * 100
        print(f"\n编码器层数 (E_Layer) 对MSE的影响:")
        print(f"  最小平均MSE: {min_e_mse:.6f}")
        print(f"  最大平均MSE: {max_e_mse:.6f}")
        print(f"  影响幅度: {e_impact:.2f}%")
        if e_impact > 5:
            print(f"  ✅ 编码器层数对MSE有显著影响")
        else:
            print(f"  ⚠️  编码器层数对MSE影响较小")
    
    # 计算D_Layer的MSE变化
    d_layer_mses = {k: np.mean([s['mse'] for s in v]) for k, v in d_layer_stats.items()}
    if len(d_layer_mses) > 1:
        min_d_mse = min(d_layer_mses.values())
        max_d_mse = max(d_layer_mses.values())
        d_impact = ((max_d_mse - min_d_mse) / min_d_mse) * 100
        print(f"\n解码器层数 (D_Layer) 对MSE的影响:")
        print(f"  最小平均MSE: {min_d_mse:.6f}")
        print(f"  最大平均MSE: {max_d_mse:.6f}")
        print(f"  影响幅度: {d_impact:.2f}%")
        if d_impact > 5:
            print(f"  ✅ 解码器层数对MSE有显著影响")
        else:
            print(f"  ⚠️  解码器层数对MSE影响较小")
    
    # 与基准对比（E_Layer=1, D_Layer=1）
    baseline_key = (1, 1)
    if baseline_key in layer_combo_stats:
        baseline_mse = np.mean([s['mse'] for s in layer_combo_stats[baseline_key]])
        best_mse = sorted_results[0].get('test_mse', float('inf'))
        improvement = ((baseline_mse - best_mse) / baseline_mse) * 100
        print(f"\n与基准 (E_Layer=1, D_Layer=1) 对比:")
        print(f"  基准MSE: {baseline_mse:.6f}")
        print(f"  最佳MSE: {best_mse:.6f}")
        if improvement > 0:
            print(f"  改进幅度: {improvement:.2f}% (降低)")
        else:
            print(f"  变化幅度: {abs(improvement):.2f}% (增加)")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析编码器和解码器层数寻优实验结果')
    parser.add_argument('--result_file', type=str, default=None, help='结果文件路径（默认: experiment_results.log）')
    parser.add_argument('--seed', type=int, default=2088, help='随机种子')
    parser.add_argument('--model_id_prefix', type=str, default='T3Time_FreTS_Gated_Qwen_LayerSearch', 
                       help='模型ID前缀')
    
    args = parser.parse_args()
    
    results = load_layer_search_results(args.result_file, args.seed, args.model_id_prefix)
    
    if not results:
        print(f"\n❌ 未找到 seed={args.seed} 的层数寻优实验结果")
        print("请先运行层数寻优脚本: bash scripts/T3Time_FreTS_FusionExp/hyperopt_layer_search.sh")
        return
    
    sorted_results, e_layer_stats, d_layer_stats, layer_combo_stats = analyze_layer_impact(results)
    print_analysis(sorted_results, e_layer_stats, d_layer_stats, layer_combo_stats)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
