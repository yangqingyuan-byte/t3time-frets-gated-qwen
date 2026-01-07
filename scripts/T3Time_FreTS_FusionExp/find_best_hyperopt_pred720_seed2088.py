#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索 T3Time_FreTS_Gated_Qwen_Hyperopt 参数寻优实验的最佳参数组合
针对 pred_len=720, seed=2088
分别找出 MSE 和 MAE 最好的参数组合
"""
import json
import os
import sys
from collections import defaultdict

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_hyperopt_results(result_file=None, seed=2088, pred_len=720, model_id_prefix="T3Time_FreTS_Gated_Qwen_Hyperopt"):
    """加载参数寻优实验结果"""
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
                # 检查是否是参数寻优实验结果
                if (data.get('seed') == seed and 
                    data.get('pred_len') == pred_len and
                    data.get('model_id', '').startswith(model_id_prefix)):
                    results.append(data)
            except json.JSONDecodeError as e:
                continue
            except Exception as e:
                continue
    
    return results

def find_best_params(results):
    """找出最佳参数组合（分别按MSE和MAE）"""
    if not results:
        return None, None, [], [], {}
    
    # 按 MSE 排序
    sorted_results_mse = sorted(results, key=lambda x: x.get('test_mse', float('inf')))
    best_mse = sorted_results_mse[0] if sorted_results_mse else None
    
    # 按 MAE 排序
    sorted_results_mae = sorted(results, key=lambda x: x.get('test_mae', float('inf')))
    best_mae = sorted_results_mae[0] if sorted_results_mae else None
    
    # 统计每个参数组合的MSE和MAE
    param_stats_mse = defaultdict(list)
    param_stats_mae = defaultdict(list)
    for r in results:
        param_key = (r.get('channel'), r.get('dropout_n'), r.get('head'))
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
    
    return best_mse, best_mae, sorted_results_mse, sorted_results_mae, param_avg

def print_results(best_mse, best_mae, sorted_results_mse, sorted_results_mae, param_avg):
    """打印结果"""
    print("="*80)
    print("T3Time_FreTS_Gated_Qwen_Hyperopt 参数寻优结果分析")
    print("Pred_Len: 720, Seed: 2088")
    print("="*80)
    
    if not best_mse or not best_mae:
        print("\n❌ 未找到实验结果")
        return
    
    total_results = len(sorted_results_mse)
    print(f"\n找到 {total_results} 条实验结果\n")
    
    # 最小MSE最佳结果
    print("="*80)
    print("🏆 最小 MSE 参数组合")
    print("="*80)
    print(f"Channel:     {best_mse.get('channel', 'N/A')}")
    print(f"Dropout:     {best_mse.get('dropout_n', 'N/A')}")
    print(f"Head:        {best_mse.get('head', 'N/A')}")
    print(f"MSE:         {best_mse.get('test_mse', 'N/A'):.6f}")
    print(f"MAE:         {best_mse.get('test_mae', 'N/A'):.6f}")
    print(f"Seed:        {best_mse.get('seed', 'N/A')}")
    print(f"Pred_Len:    {best_mse.get('pred_len', 'N/A')}")
    print(f"Timestamp:   {best_mse.get('timestamp', 'N/A')}")
    
    # 最小MAE最佳结果
    print("\n" + "="*80)
    print("🏆 最小 MAE 参数组合")
    print("="*80)
    print(f"Channel:     {best_mae.get('channel', 'N/A')}")
    print(f"Dropout:     {best_mae.get('dropout_n', 'N/A')}")
    print(f"Head:        {best_mae.get('head', 'N/A')}")
    print(f"MSE:         {best_mae.get('test_mse', 'N/A'):.6f}")
    print(f"MAE:         {best_mae.get('test_mae', 'N/A'):.6f}")
    print(f"Seed:        {best_mae.get('seed', 'N/A')}")
    print(f"Pred_Len:    {best_mae.get('pred_len', 'N/A')}")
    print(f"Timestamp:   {best_mae.get('timestamp', 'N/A')}")
    
    # Top 10 最佳结果（按MSE）
    print("\n" + "="*80)
    print("Top 10 最佳配置（按 MSE 排序）")
    print("="*80)
    print(f"{'Rank':<6} {'Channel':<10} {'Dropout':<10} {'Head':<8} {'MSE':<15} {'MAE':<15}")
    print("-"*80)
    
    for i, r in enumerate(sorted_results_mse[:10], 1):
        print(f"{i:<6} {r.get('channel', 'N/A'):<10} {r.get('dropout_n', 'N/A'):<10.2f} "
              f"{r.get('head', 'N/A'):<8} {r.get('test_mse', 'N/A'):<15.6f} {r.get('test_mae', 'N/A'):<15.6f}")
    
    # Top 10 最佳结果（按MAE）
    print("\n" + "="*80)
    print("Top 10 最佳配置（按 MAE 排序）")
    print("="*80)
    print(f"{'Rank':<6} {'Channel':<10} {'Dropout':<10} {'Head':<8} {'MSE':<15} {'MAE':<15}")
    print("-"*80)
    
    for i, r in enumerate(sorted_results_mae[:10], 1):
        print(f"{i:<6} {r.get('channel', 'N/A'):<10} {r.get('dropout_n', 'N/A'):<10.2f} "
              f"{r.get('head', 'N/A'):<8} {r.get('test_mse', 'N/A'):<15.6f} {r.get('test_mae', 'N/A'):<15.6f}")
    
    # 参数统计分析（按MSE）
    print("\n" + "="*80)
    print("参数统计分析（按平均 MSE 排序）")
    print("="*80)
    print(f"{'Channel':<10} {'Dropout':<10} {'Head':<8} {'平均MSE':<15} {'最小MSE':<15} {'最大MSE':<15} {'次数':<8}")
    print("-"*80)
    
    sorted_params_mse = sorted(param_avg.items(), key=lambda x: x[1]['mse_mean'])
    for (channel, dropout, head), stats in sorted_params_mse[:20]:  # 显示前20个
        print(f"{channel:<10} {dropout:<10.2f} {head:<8} "
              f"{stats['mse_mean']:<15.6f} {stats['mse_min']:<15.6f} {stats['mse_max']:<15.6f} {stats['count']:<8}")
    
    # 参数统计分析（按MAE）
    print("\n" + "="*80)
    print("参数统计分析（按平均 MAE 排序）")
    print("="*80)
    print(f"{'Channel':<10} {'Dropout':<10} {'Head':<8} {'平均MAE':<15} {'最小MAE':<15} {'最大MAE':<15} {'次数':<8}")
    print("-"*80)
    
    sorted_params_mae = sorted(param_avg.items(), key=lambda x: x[1]['mae_mean'])
    for (channel, dropout, head), stats in sorted_params_mae[:20]:  # 显示前20个
        print(f"{channel:<10} {dropout:<10.2f} {head:<8} "
              f"{stats['mae_mean']:<15.6f} {stats['mae_min']:<15.6f} {stats['mae_max']:<15.6f} {stats['count']:<8}")
    
    # 各参数维度分析
    print("\n" + "="*80)
    print("各参数维度分析（MSE）")
    print("="*80)
    
    # Channel 分析（MSE）
    channel_stats_mse = defaultdict(list)
    for r in sorted_results_mse:
        channel_stats_mse[r.get('channel')].append(r.get('test_mse', float('inf')))
    
    print("\n[1] Channel 参数分析（MSE）:")
    print(f"{'Channel':<10} {'平均MSE':<15} {'最小MSE':<15} {'最大MSE':<15} {'次数':<8}")
    print("-"*80)
    for channel in sorted(channel_stats_mse.keys()):
        mse_list = channel_stats_mse[channel]
        print(f"{channel:<10} {sum(mse_list)/len(mse_list):<15.6f} "
              f"{min(mse_list):<15.6f} {max(mse_list):<15.6f} {len(mse_list):<8}")
    
    # Dropout 分析（MSE）
    dropout_stats_mse = defaultdict(list)
    for r in sorted_results_mse:
        dropout_stats_mse[r.get('dropout_n')].append(r.get('test_mse', float('inf')))
    
    print("\n[2] Dropout 参数分析（MSE）:")
    print(f"{'Dropout':<10} {'平均MSE':<15} {'最小MSE':<15} {'最大MSE':<15} {'次数':<8}")
    print("-"*80)
    for dropout in sorted(dropout_stats_mse.keys()):
        mse_list = dropout_stats_mse[dropout]
        print(f"{dropout:<10.2f} {sum(mse_list)/len(mse_list):<15.6f} "
              f"{min(mse_list):<15.6f} {max(mse_list):<15.6f} {len(mse_list):<8}")
    
    # Head 分析（MSE）
    head_stats_mse = defaultdict(list)
    for r in sorted_results_mse:
        head_stats_mse[r.get('head')].append(r.get('test_mse', float('inf')))
    
    print("\n[3] Head 参数分析（MSE）:")
    print(f"{'Head':<10} {'平均MSE':<15} {'最小MSE':<15} {'最大MSE':<15} {'次数':<8}")
    print("-"*80)
    for head in sorted(head_stats_mse.keys()):
        mse_list = head_stats_mse[head]
        print(f"{head:<10} {sum(mse_list)/len(mse_list):<15.6f} "
              f"{min(mse_list):<15.6f} {max(mse_list):<15.6f} {len(mse_list):<8}")
    
    # 各参数维度分析（MAE）
    print("\n" + "="*80)
    print("各参数维度分析（MAE）")
    print("="*80)
    
    # Channel 分析（MAE）
    channel_stats_mae = defaultdict(list)
    for r in sorted_results_mae:
        channel_stats_mae[r.get('channel')].append(r.get('test_mae', float('inf')))
    
    print("\n[1] Channel 参数分析（MAE）:")
    print(f"{'Channel':<10} {'平均MAE':<15} {'最小MAE':<15} {'最大MAE':<15} {'次数':<8}")
    print("-"*80)
    for channel in sorted(channel_stats_mae.keys()):
        mae_list = channel_stats_mae[channel]
        print(f"{channel:<10} {sum(mae_list)/len(mae_list):<15.6f} "
              f"{min(mae_list):<15.6f} {max(mae_list):<15.6f} {len(mae_list):<8}")
    
    # Dropout 分析（MAE）
    dropout_stats_mae = defaultdict(list)
    for r in sorted_results_mae:
        dropout_stats_mae[r.get('dropout_n')].append(r.get('test_mae', float('inf')))
    
    print("\n[2] Dropout 参数分析（MAE）:")
    print(f"{'Dropout':<10} {'平均MAE':<15} {'最小MAE':<15} {'最大MAE':<15} {'次数':<8}")
    print("-"*80)
    for dropout in sorted(dropout_stats_mae.keys()):
        mae_list = dropout_stats_mae[dropout]
        print(f"{dropout:<10.2f} {sum(mae_list)/len(mae_list):<15.6f} "
              f"{min(mae_list):<15.6f} {max(mae_list):<15.6f} {len(mae_list):<8}")
    
    # Head 分析（MAE）
    head_stats_mae = defaultdict(list)
    for r in sorted_results_mae:
        head_stats_mae[r.get('head')].append(r.get('test_mae', float('inf')))
    
    print("\n[3] Head 参数分析（MAE）:")
    print(f"{'Head':<10} {'平均MAE':<15} {'最小MAE':<15} {'最大MAE':<15} {'次数':<8}")
    print("-"*80)
    for head in sorted(head_stats_mae.keys()):
        mae_list = head_stats_mae[head]
        print(f"{head:<10} {sum(mae_list)/len(mae_list):<15.6f} "
              f"{min(mae_list):<15.6f} {max(mae_list):<15.6f} {len(mae_list):<8}")
    
    # 最佳参数组合（按平均MSE）
    if param_avg:
        best_param_avg_mse = min(param_avg.items(), key=lambda x: x[1]['mse_mean'])
        (best_channel_mse, best_dropout_mse, best_head_mse), best_stats_mse = best_param_avg_mse
        
        print("\n" + "="*80)
        print("🏆 最佳参数组合（按平均 MSE）")
        print("="*80)
        print(f"Channel:     {best_channel_mse}")
        print(f"Dropout:     {best_dropout_mse}")
        print(f"Head:        {best_head_mse}")
        print(f"平均 MSE:    {best_stats_mse['mse_mean']:.6f}")
        print(f"最小 MSE:    {best_stats_mse['mse_min']:.6f}")
        print(f"最大 MSE:    {best_stats_mse['mse_max']:.6f}")
        print(f"平均 MAE:    {best_stats_mse['mae_mean']:.6f}")
        print(f"最小 MAE:    {best_stats_mse['mae_min']:.6f}")
        print(f"最大 MAE:    {best_stats_mse['mae_max']:.6f}")
        print(f"实验次数:    {best_stats_mse['count']}")
        
        # 最佳参数组合（按平均MAE）
        best_param_avg_mae = min(param_avg.items(), key=lambda x: x[1]['mae_mean'])
        (best_channel_mae, best_dropout_mae, best_head_mae), best_stats_mae = best_param_avg_mae
        
        print("\n" + "="*80)
        print("🏆 最佳参数组合（按平均 MAE）")
        print("="*80)
        print(f"Channel:     {best_channel_mae}")
        print(f"Dropout:     {best_dropout_mae}")
        print(f"Head:        {best_head_mae}")
        print(f"平均 MSE:    {best_stats_mae['mse_mean']:.6f}")
        print(f"最小 MSE:    {best_stats_mae['mse_min']:.6f}")
        print(f"最大 MSE:    {best_stats_mae['mse_max']:.6f}")
        print(f"平均 MAE:    {best_stats_mae['mae_mean']:.6f}")
        print(f"最小 MAE:    {best_stats_mae['mae_min']:.6f}")
        print(f"最大 MAE:    {best_stats_mae['mae_max']:.6f}")
        print(f"实验次数:    {best_stats_mae['count']}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检索 T3Time_FreTS_Gated_Qwen_Hyperopt 参数寻优实验的最佳参数组合')
    parser.add_argument('--result_file', type=str, default=None, help='结果文件路径（默认: experiment_results.log）')
    parser.add_argument('--seed', type=int, default=2088, help='随机种子')
    parser.add_argument('--pred_len', type=int, default=720, help='预测长度')
    parser.add_argument('--model_id_prefix', type=str, default='T3Time_FreTS_Gated_Qwen_Hyperopt', 
                       help='模型ID前缀')
    
    args = parser.parse_args()
    
    results = load_hyperopt_results(args.result_file, args.seed, args.pred_len, args.model_id_prefix)
    
    if not results:
        print(f"\n❌ 未找到 seed={args.seed}, pred_len={args.pred_len} 的参数寻优实验结果")
        print("请先运行参数寻优脚本: bash scripts/T3Time_FreTS_FusionExp/hyperopt_pred720_seed2088.sh")
        return
    
    best_mse, best_mae, sorted_results_mse, sorted_results_mae, param_avg = find_best_params(results)
    print_results(best_mse, best_mae, sorted_results_mse, sorted_results_mae, param_avg)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
