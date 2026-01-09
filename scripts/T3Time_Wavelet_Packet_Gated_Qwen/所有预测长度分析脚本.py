#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索 T3Time_Wavelet_Packet_Gated_Qwen 模型的所有种子的所有配置实验结果
按预测长度（96, 192, 336, 720）分别分析
支持分析所有种子或指定种子的实验结果
"""
import json
import os
import sys
from collections import defaultdict

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_hyperopt_results(result_file=None, seed=None, model_id_prefix="T3Time_Wavelet_Packet_Gated_Qwen"):
    """
    加载参数寻优实验结果
    
    Args:
        result_file: 结果文件路径，默认为 experiment_results.log
        seed: 随机种子，如果为 None 则加载所有种子的结果
        model_id_prefix: 模型名称前缀
    """
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
                # 检查是否是目标模型的实验结果
                if data.get('model', '').startswith(model_id_prefix):
                    # 如果指定了 seed，则只加载该 seed 的结果；否则加载所有 seed
                    if seed is None or data.get('seed') == seed:
                        results.append(data)
            except json.JSONDecodeError as e:
                continue
            except Exception as e:
                continue
    
    return results

def find_best_params_by_pred_len(results, pred_lens=[96, 192, 336, 720]):
    """按预测长度分组，找出每个预测长度的最佳参数组合"""
    if not results:
        return {}
    
    results_by_pred_len = {}
    
    for pred_len in pred_lens:
        # 筛选该预测长度的结果
        pred_results = [r for r in results if r.get('pred_len') == pred_len]
        
        if not pred_results:
            results_by_pred_len[pred_len] = {
                'best_mse': None,
                'best_mae': None,
                'sorted_results_mse': [],
                'sorted_results_mae': [],
                'param_avg': {},
                'count': 0
            }
            continue
        
        # 按 MSE 排序
        sorted_results_mse = sorted(pred_results, key=lambda x: x.get('test_mse', float('inf')))
        best_mse = sorted_results_mse[0] if sorted_results_mse else None
        
        # 按 MAE 排序
        sorted_results_mae = sorted(pred_results, key=lambda x: x.get('test_mae', float('inf')))
        best_mae = sorted_results_mae[0] if sorted_results_mae else None
        
        # 统计每个参数组合的MSE和MAE
        param_stats_mse = defaultdict(list)
        param_stats_mae = defaultdict(list)
        for r in pred_results:
            # 对于 Wavelet Packet 模型，关键参数包括 channel, dropout_n, head, wp_level, wavelet
            param_key = (
                r.get('channel'), 
                r.get('dropout_n'), 
                r.get('head'),
                r.get('wp_level', 'N/A')
            )
            param_stats_mse[param_key].append(r.get('test_mse', float('inf')))
            param_stats_mae[param_key].append(r.get('test_mae', float('inf')))
        
        # 计算每个参数组合的平均 MSE 和 MAE
        param_avg = {}
        for param_key in param_stats_mse.keys():
            mse_list = param_stats_mse[param_key]
            mae_list = param_stats_mae[param_key]
            param_avg[param_key] = {
                'mse_mean': sum(mse_list) / len(mse_list),
                'mse_min': min(mse_list),
                'mse_max': max(mse_list),
                'mae_mean': sum(mae_list) / len(mae_list),
                'mae_min': min(mae_list),
                'mae_max': max(mae_list),
                'count': len(mse_list)
            }
        
        results_by_pred_len[pred_len] = {
            'best_mse': best_mse,
            'best_mae': best_mae,
            'sorted_results_mse': sorted_results_mse,
            'sorted_results_mae': sorted_results_mae,
            'param_avg': param_avg,
            'count': len(pred_results)
        }
    
    return results_by_pred_len

def get_seed_statistics(results):
    """统计所有结果的种子分布"""
    seed_counts = defaultdict(int)
    seed_by_pred_len = defaultdict(lambda: defaultdict(int))
    
    for r in results:
        seed = r.get('seed', 'Unknown')
        pred_len = r.get('pred_len', 'Unknown')
        seed_counts[seed] += 1
        seed_by_pred_len[pred_len][seed] += 1
    
    return seed_counts, seed_by_pred_len

def print_results_by_pred_len(results_by_pred_len, pred_lens=[96, 192, 336, 720], all_results=None):
    """按预测长度打印结果"""
    print("="*80)
    print("T3Time_Wavelet_Packet_Gated_Qwen 参数寻优结果分析（所有种子）")
    print("按预测长度分别分析: {}".format(", ".join(map(str, pred_lens))))
    print("="*80)
    
    # 统计总结果数和种子分布
    total_results = sum(data['count'] for data in results_by_pred_len.values())
    
    if all_results:
        seed_counts, seed_by_pred_len_stats = get_seed_statistics(all_results)
        print(f"\n找到 {total_results} 条实验结果")
        print(f"涉及 {len(seed_counts)} 个不同的种子: {sorted(seed_counts.keys())}")
        print("\n种子分布统计:")
        print(f"{'Seed':<10} {'总实验数':<12}")
        print("-"*25)
        for seed in sorted(seed_counts.keys()):
            print(f"{seed:<10} {seed_counts[seed]:<12}")
    else:
        print(f"\n找到 {total_results} 条实验结果\n")
    
    # 对每个预测长度分别分析
    for pred_len in pred_lens:
        data = results_by_pred_len.get(pred_len, {})
        best_mse = data.get('best_mse')
        best_mae = data.get('best_mae')
        sorted_results_mse = data.get('sorted_results_mse', [])
        sorted_results_mae = data.get('sorted_results_mae', [])
        param_avg = data.get('param_avg', {})
        count = data.get('count', 0)
        
        if not best_mse or not best_mae:
            print("\n" + "="*80)
            print(f"预测长度 {pred_len}: 未找到实验结果")
            print("="*80)
            continue
        
        # 简化输出，不显示实验数量
        
        # 打印该预测长度的最佳结果
        print_single_pred_len_results(best_mse, best_mae, sorted_results_mse, sorted_results_mae, param_avg, pred_len)

def print_single_pred_len_results(best_mse, best_mae, sorted_results_mse, sorted_results_mae, param_avg, pred_len):
    """打印单个预测长度的结果（简化版：只显示关键信息）"""
    
    print("\n" + "="*80)
    print(f"📊 预测长度 {pred_len}")
    print("="*80)
    
    # 1. 最佳MSE结果
    print("\n【最佳 MSE】")
    print(f"  MSE: {best_mse.get('test_mse', 'N/A'):.6f}")
    print(f"  MAE: {best_mse.get('test_mae', 'N/A'):.6f}")
    print(f"  Seed: {best_mse.get('seed', 'N/A')}")
    print(f"  Channel: {best_mse.get('channel', 'N/A')}, Dropout: {best_mse.get('dropout_n', 'N/A')}, "
          f"Head: {best_mse.get('head', 'N/A')}, WP_Level: {best_mse.get('wp_level', 'N/A')}")
    
    # 2. 最佳MAE结果
    print("\n【最佳 MAE】")
    print(f"  MSE: {best_mae.get('test_mse', 'N/A'):.6f}")
    print(f"  MAE: {best_mae.get('test_mae', 'N/A'):.6f}")
    print(f"  Seed: {best_mae.get('seed', 'N/A')}")
    print(f"  Channel: {best_mae.get('channel', 'N/A')}, Dropout: {best_mae.get('dropout_n', 'N/A')}, "
          f"Head: {best_mae.get('head', 'N/A')}, WP_Level: {best_mae.get('wp_level', 'N/A')}")
    
    # 3. 最佳参数组合（按平均MSE）
    if param_avg:
        best_param_avg_mse = min(param_avg.items(), key=lambda x: x[1]['mse_mean'])
        (best_channel_mse, best_dropout_mse, best_head_mse, best_wp_level_mse), best_stats_mse = best_param_avg_mse
        
        print("\n【最佳参数组合（按平均 MSE）】")
        print(f"  平均 MSE: {best_stats_mse['mse_mean']:.6f}")
        print(f"  平均 MAE: {best_stats_mse['mae_mean']:.6f}")
        print(f"  Channel: {best_channel_mse if best_channel_mse is not None else 'N/A'}, "
              f"Dropout: {best_dropout_mse if best_dropout_mse is not None else 'N/A'}, "
              f"Head: {best_head_mse if best_head_mse is not None else 'N/A'}, "
              f"WP_Level: {best_wp_level_mse if best_wp_level_mse is not None else 'N/A'}")
        print(f"  实验次数: {best_stats_mse['count']}")
        
        # 4. 最佳参数组合（按平均MAE）
        best_param_avg_mae = min(param_avg.items(), key=lambda x: x[1]['mae_mean'])
        (best_channel_mae, best_dropout_mae, best_head_mae, best_wp_level_mae), best_stats_mae = best_param_avg_mae
        
        print("\n【最佳参数组合（按平均 MAE）】")
        print(f"  平均 MSE: {best_stats_mae['mse_mean']:.6f}")
        print(f"  平均 MAE: {best_stats_mae['mae_mean']:.6f}")
        print(f"  Channel: {best_channel_mae if best_channel_mae is not None else 'N/A'}, "
              f"Dropout: {best_dropout_mae if best_dropout_mae is not None else 'N/A'}, "
              f"Head: {best_head_mae if best_head_mae is not None else 'N/A'}, "
              f"WP_Level: {best_wp_level_mae if best_wp_level_mae is not None else 'N/A'}")
        print(f"  实验次数: {best_stats_mae['count']}")

def print_summary_table(results_by_pred_len, pred_lens=[96, 192, 336, 720]):
    """打印所有预测长度的汇总表格"""
    print("\n" + "="*80)
    print("📊 所有预测长度的最佳结果汇总（跨所有种子）")
    print("="*80)
    
    # MSE 汇总（添加综合均值）
    print("\n【最小 MSE 汇总】")
    print(f"{'Pred_Len':<12} {'Seed':<8} {'Channel':<10} {'Dropout':<10} {'Head':<8} {'WP_Level':<10} {'MSE':<15} {'MAE':<15}")
    print("-"*110)
    
    mse_values = []
    mae_values = []
    
    for pred_len in pred_lens:
        data = results_by_pred_len.get(pred_len, {})
        best_mse = data.get('best_mse')
        
        if best_mse:
            wp_level = best_mse.get('wp_level', 'N/A')
            seed = best_mse.get('seed', 'N/A')
            mse_val = best_mse.get('test_mse')
            mae_val = best_mse.get('test_mae')
            
            if mse_val is not None:
                mse_values.append(mse_val)
            if mae_val is not None:
                mae_values.append(mae_val)
            
            print(f"{pred_len:<12} {seed:<8} {best_mse.get('channel', 'N/A'):<10} "
                  f"{best_mse.get('dropout_n', 'N/A'):<10.1f} {best_mse.get('head', 'N/A'):<8} {wp_level:<10} "
                  f"{mse_val:<15.6f} {mae_val:<15.6f}")
        else:
            print(f"{pred_len:<12} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<10} "
                  f"{'N/A':<15} {'N/A':<15}")
    
    # 显示综合均值
    if mse_values and mae_values:
        mse_avg = sum(mse_values) / len(mse_values)
        mae_avg = sum(mae_values) / len(mae_values)
        print("-"*110)
        print(f"{'综合均值':<12} {'':<8} {'':<10} {'':<10} {'':<8} {'':<10} "
              f"{mse_avg:<15.6f} {mae_avg:<15.6f}")
    
    # MAE 汇总（添加综合均值）
    print("\n【最小 MAE 汇总】")
    print(f"{'Pred_Len':<12} {'Seed':<8} {'Channel':<10} {'Dropout':<10} {'Head':<8} {'WP_Level':<10} {'MSE':<15} {'MAE':<15}")
    print("-"*110)
    
    mse_values_mae = []
    mae_values_mae = []
    
    for pred_len in pred_lens:
        data = results_by_pred_len.get(pred_len, {})
        best_mae = data.get('best_mae')
        
        if best_mae:
            wp_level = best_mae.get('wp_level', 'N/A')
            seed = best_mae.get('seed', 'N/A')
            mse_val = best_mae.get('test_mse')
            mae_val = best_mae.get('test_mae')
            
            if mse_val is not None:
                mse_values_mae.append(mse_val)
            if mae_val is not None:
                mae_values_mae.append(mae_val)
            
            print(f"{pred_len:<12} {seed:<8} {best_mae.get('channel', 'N/A'):<10} "
                  f"{best_mae.get('dropout_n', 'N/A'):<10.1f} {best_mae.get('head', 'N/A'):<8} {wp_level:<10} "
                  f"{mse_val:<15.6f} {mae_val:<15.6f}")
        else:
            print(f"{pred_len:<12} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<10} "
                  f"{'N/A':<15} {'N/A':<15}")
    
    # 显示综合均值
    if mse_values_mae and mae_values_mae:
        mse_avg_mae = sum(mse_values_mae) / len(mse_values_mae)
        mae_avg_mae = sum(mae_values_mae) / len(mae_values_mae)
        print("-"*110)
        print(f"{'综合均值':<12} {'':<8} {'':<10} {'':<10} {'':<8} {'':<10} "
              f"{mse_avg_mae:<15.6f} {mae_avg_mae:<15.6f}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检索 T3Time_Wavelet_Packet_Gated_Qwen 模型的所有种子的参数寻优实验结果（按预测长度分别分析）')
    parser.add_argument('--result_file', type=str, default=None, help='结果文件路径（默认: experiment_results.log）')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（默认: None，分析所有种子）')
    parser.add_argument('--model_id_prefix', type=str, default='T3Time_Wavelet_Packet_Gated_Qwen', 
                       help='模型名称前缀')
    parser.add_argument('--pred_lens', type=int, nargs='+', default=[96, 192, 336, 720],
                       help='要分析的预测长度列表（默认: 96 192 336 720）')
    
    args = parser.parse_args()
    
    results = load_hyperopt_results(args.result_file, args.seed, args.model_id_prefix)
    
    if not results:
        if args.seed is None:
            print(f"\n❌ 未找到 {args.model_id_prefix} 模型的任何实验结果")
        else:
            print(f"\n❌ 未找到 seed={args.seed} 的参数寻优实验结果")
        print("请先运行参数寻优脚本进行实验")
        return
    
    # 按预测长度分组分析
    results_by_pred_len = find_best_params_by_pred_len(results, args.pred_lens)
    
    # 打印汇总表格
    print_summary_table(results_by_pred_len, args.pred_lens)
    
    # # 打印每个预测长度的详细结果（传入所有结果用于种子统计）
    # print_results_by_pred_len(results_by_pred_len, args.pred_lens, all_results=results)
    
    # print("\n" + "="*80)
    # print("分析完成！")
    # print("="*80)

if __name__ == "__main__":
    main()
