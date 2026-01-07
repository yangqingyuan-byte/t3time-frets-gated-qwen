#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析使用最佳参数组合在多种子（2020-2090）上的训练结果
找出是否有更小的MSE结果
"""
import json
import os
import sys
from collections import defaultdict
from tabulate import tabulate

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_multi_seed_results(result_file=None, model_id_prefix="T3Time_FreTS_Gated_Qwen_BestParams_MultiSeed", 
                           model_name=None, pred_lens=None):
    """加载多种子训练结果，支持多个预测长度
    
    Args:
        result_file: 结果文件路径
        model_id_prefix: model_id前缀（优先匹配）
        model_name: 模型名称（如果model_id_prefix匹配不到，则使用model_name匹配）
        pred_lens: 预测长度列表
    """
    if result_file is None:
        result_file = os.path.join(project_root, "experiment_results.log")
    
    if pred_lens is None:
        pred_lens = [96, 720]
    
    # 将pred_lens转换为列表（如果传入的是单个值）
    if isinstance(pred_lens, int):
        pred_lens = [pred_lens]
    
    # 如果没有指定model_name，从model_id_prefix推断
    if model_name is None:
        # 从model_id_prefix推断模型名称
        # 例如: "T3Time_FreTS_Gated_Qwen_BestParams_MultiSeed" -> "T3Time_FreTS_Gated_Qwen"
        if "T3Time_FreTS_Gated_Qwen" in model_id_prefix:
            model_name = "T3Time_FreTS_Gated_Qwen"
    
    results_by_pred_len = {pred_len: [] for pred_len in pred_lens}
    
    if not os.path.exists(result_file):
        print(f"❌ 结果文件不存在: {result_file}")
        return results_by_pred_len
    
    with open(result_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                pred_len = data.get('pred_len')
                
                if pred_len not in pred_lens:
                    continue
                
                # 优先匹配model_id_prefix
                matched = False
                if model_id_prefix:
                    model_id = data.get('model_id', '')
                    if model_id.startswith(model_id_prefix):
                        matched = True
                
                # 如果model_id_prefix没匹配到，尝试匹配model_name
                if not matched and model_name:
                    model = data.get('model', '')
                    if model == model_name:
                        matched = True
                
                if matched:
                    results_by_pred_len[pred_len].append(data)
            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue
    
    return results_by_pred_len

def analyze_results_for_pred_len(results, pred_len, baseline_mse=None, baseline_mae=None):
    """针对单个预测长度分析并打印结果"""
    print("="*80)
    print(f"最佳参数组合多种子训练结果分析 - Pred_Len: {pred_len}")
    print("="*80)
    
    if not results:
        print(f"\n❌ 未找到 Pred_Len={pred_len} 的实验结果")
        return
    
    print(f"\n找到 {len(results)} 条 Pred_Len={pred_len} 的实验结果\n")
    
    # 按MSE排序
    sorted_results_mse = sorted(results, key=lambda x: x.get('test_mse', float('inf')))
    # 按MAE排序
    sorted_results_mae = sorted(results, key=lambda x: x.get('test_mae', float('inf')))
    
    # 最佳结果（最小MSE）
    best_mse_result = sorted_results_mse[0] if sorted_results_mse else None
    # 最佳结果（最小MAE）
    best_mae_result = sorted_results_mae[0] if sorted_results_mae else None
    
    # 基准值（如果没有提供，使用默认值）
    if baseline_mse is None:
        baseline_mse = 0.462425 if pred_len == 720 else None
    if baseline_mae is None:
        baseline_mae = 0.458175 if pred_len == 720 else None
    
    # 显示最小MSE结果
    if best_mse_result:
        print("="*80)
        print("🏆 最佳结果（最小MSE）")
        print("="*80)
        print(f"Seed:        {best_mse_result.get('seed', 'N/A')}")
        print(f"MSE:         {best_mse_result.get('test_mse', 'N/A'):.6f}")
        print(f"MAE:         {best_mse_result.get('test_mae', 'N/A'):.6f}")
        print(f"Pred_Len:    {best_mse_result.get('pred_len', 'N/A')}")
        print(f"Timestamp:   {best_mse_result.get('timestamp', 'N/A')}")
        print(f"\n参数配置:")
        print(f"  Channel:     {best_mse_result.get('channel', 'N/A')}")
        print(f"  Dropout:     {best_mse_result.get('dropout_n', 'N/A')}")
        print(f"  Head:        {best_mse_result.get('head', 'N/A')}")
        print(f"  Batch Size:  {best_mse_result.get('batch_size', 'N/A')}")
        print(f"  Learning Rate: {best_mse_result.get('learning_rate', 'N/A')}")
        print(f"  Weight Decay:  {best_mse_result.get('weight_decay', 'N/A')}")
        print(f"  Loss Function: {best_mse_result.get('loss_fn', 'N/A')}")
        
        # 与基准对比（MSE）
        if baseline_mse is not None:
            improvement_mse = (baseline_mse - best_mse_result.get('test_mse', baseline_mse)) / baseline_mse * 100
            print(f"\n与基准对比 (seed=2088, MSE={baseline_mse:.6f}):")
            if improvement_mse > 0:
                print(f"  ✅ 改进: {improvement_mse:.2f}% (MSE降低)")
            elif improvement_mse < 0:
                print(f"  ⚠️  退步: {abs(improvement_mse):.2f}% (MSE升高)")
            else:
                print(f"  ➡️  持平")
    
    # 显示最小MAE结果
    if best_mae_result:
        print("\n" + "="*80)
        print("🏆 最佳结果（最小MAE）")
        print("="*80)
        print(f"Seed:        {best_mae_result.get('seed', 'N/A')}")
        print(f"MSE:         {best_mae_result.get('test_mse', 'N/A'):.6f}")
        print(f"MAE:         {best_mae_result.get('test_mae', 'N/A'):.6f}")
        print(f"Pred_Len:    {best_mae_result.get('pred_len', 'N/A')}")
        print(f"Timestamp:   {best_mae_result.get('timestamp', 'N/A')}")
        print(f"\n参数配置:")
        print(f"  Channel:     {best_mae_result.get('channel', 'N/A')}")
        print(f"  Dropout:     {best_mae_result.get('dropout_n', 'N/A')}")
        print(f"  Head:        {best_mae_result.get('head', 'N/A')}")
        print(f"  Batch Size:  {best_mae_result.get('batch_size', 'N/A')}")
        print(f"  Learning Rate: {best_mae_result.get('learning_rate', 'N/A')}")
        print(f"  Weight Decay:  {best_mae_result.get('weight_decay', 'N/A')}")
        print(f"  Loss Function: {best_mae_result.get('loss_fn', 'N/A')}")
        
        # 与基准对比（MAE）
        if baseline_mae is not None:
            improvement_mae = (baseline_mae - best_mae_result.get('test_mae', baseline_mae)) / baseline_mae * 100
            print(f"\n与基准对比 (seed=2088, MAE={baseline_mae:.6f}):")
            if improvement_mae > 0:
                print(f"  ✅ 改进: {improvement_mae:.2f}% (MAE降低)")
            elif improvement_mae < 0:
                print(f"  ⚠️  退步: {abs(improvement_mae):.2f}% (MAE升高)")
            else:
                print(f"  ➡️  持平")
        
        # 检查最小MSE和最小MAE是否是同一个结果
        if best_mse_result and best_mae_result:
            if best_mse_result.get('seed') == best_mae_result.get('seed'):
                print(f"\n  💡 注意: 最小MSE和最小MAE来自同一个种子 ({best_mse_result.get('seed')})")
            else:
                print(f"\n  💡 注意: 最小MSE和最小MAE来自不同的种子")
                print(f"     最小MSE种子: {best_mse_result.get('seed')}, 最小MAE种子: {best_mae_result.get('seed')}")
    
    # Top 10 最佳结果（按MSE排序）
    print("\n" + "="*80)
    print("Top 10 最佳结果（按MSE排序）")
    print("="*80)
    
    table_headers_mse = ["排名", "Seed", "MSE", "MAE", "MSE改进幅度"]
    table_data_mse = []
    
    for i, r in enumerate(sorted_results_mse[:10], 1):
        mse = r.get('test_mse', float('inf'))
        mae = r.get('test_mae', float('inf'))
        seed = r.get('seed', 'N/A')
        
        if baseline_mse is not None and baseline_mse > 0:
            improvement = (baseline_mse - mse) / baseline_mse * 100
            improvement_str = f"{improvement:+.2f}%"
        else:
            improvement_str = "N/A"
        
        table_data_mse.append([
            i,
            seed,
            f"{mse:.6f}",
            f"{mae:.6f}",
            improvement_str
        ])
    
    print(tabulate(table_data_mse, headers=table_headers_mse, tablefmt="grid"))
    
    # Top 10 最佳结果（按MAE排序）
    print("\n" + "="*80)
    print("Top 10 最佳结果（按MAE排序）")
    print("="*80)
    
    table_headers_mae = ["排名", "Seed", "MSE", "MAE", "MAE改进幅度"]
    table_data_mae = []
    
    for i, r in enumerate(sorted_results_mae[:10], 1):
        mse = r.get('test_mse', float('inf'))
        mae = r.get('test_mae', float('inf'))
        seed = r.get('seed', 'N/A')
        
        if baseline_mae is not None and baseline_mae > 0:
            improvement = (baseline_mae - mae) / baseline_mae * 100
            improvement_str = f"{improvement:+.2f}%"
        else:
            improvement_str = "N/A"
        
        table_data_mae.append([
            i,
            seed,
            f"{mse:.6f}",
            f"{mae:.6f}",
            improvement_str
        ])
    
    print(tabulate(table_data_mae, headers=table_headers_mae, tablefmt="grid"))
    
    # 统计信息
    print("\n" + "="*80)
    print("统计信息")
    print("="*80)
    
    mse_list = [r.get('test_mse', float('inf')) for r in results]
    mae_list = [r.get('test_mae', float('inf')) for r in results]
    
    print(f"总实验数:     {len(results)}")
    print(f"MSE 统计:")
    print(f"  平均:       {sum(mse_list) / len(mse_list):.6f}")
    print(f"  最小:       {min(mse_list):.6f}")
    print(f"  最大:       {max(mse_list):.6f}")
    print(f"  中位数:     {sorted(mse_list)[len(mse_list)//2]:.6f}")
    print(f"  标准差:     {(sum((x - sum(mse_list)/len(mse_list))**2 for x in mse_list) / len(mse_list))**0.5:.6f}")
    
    print(f"\nMAE 统计:")
    print(f"  平均:       {sum(mae_list) / len(mae_list):.6f}")
    print(f"  最小:       {min(mae_list):.6f}")
    print(f"  最大:       {max(mae_list):.6f}")
    print(f"  中位数:     {sorted(mae_list)[len(mae_list)//2]:.6f}")
    print(f"  标准差:     {(sum((x - sum(mae_list)/len(mae_list))**2 for x in mae_list) / len(mae_list))**0.5:.6f}")
    
    # 优于基准的结果数量
    if baseline_mse is not None and baseline_mse > 0:
        better_mse_count = sum(1 for mse in mse_list if mse < baseline_mse)
        print(f"\n优于基准 (MSE < {baseline_mse:.6f}) 的结果数: {better_mse_count} / {len(results)} ({better_mse_count/len(results)*100:.1f}%)")
    
    if baseline_mae is not None and baseline_mae > 0:
        better_mae_count = sum(1 for mae in mae_list if mae < baseline_mae)
        print(f"优于基准 (MAE < {baseline_mae:.6f}) 的结果数: {better_mae_count} / {len(results)} ({better_mae_count/len(results)*100:.1f}%)")
    
    # 按种子范围分组统计
    print("\n" + "="*80)
    print("按种子范围分组统计")
    print("="*80)
    
    seed_ranges = [
        (2020, 2030, "2020-2030"),
        (2031, 2040, "2031-2040"),
        (2041, 2050, "2051-2050"),
        (2051, 2060, "2051-2060"),
        (2061, 2070, "2061-2070"),
        (2071, 2080, "2071-2080"),
        (2081, 2090, "2081-2090"),
    ]
    
    range_table_headers = ["种子范围", "实验数", "平均MSE", "最小MSE", "最大MSE", "平均MAE", "最小MAE", "最大MAE"]
    range_table_data = []
    
    for start, end, label in seed_ranges:
        range_results = [r for r in results if start <= r.get('seed', 0) <= end]
        if range_results:
            range_mse_list = [r.get('test_mse', float('inf')) for r in range_results]
            range_mae_list = [r.get('test_mae', float('inf')) for r in range_results]
            range_table_data.append([
                label,
                len(range_results),
                f"{sum(range_mse_list) / len(range_mse_list):.6f}",
                f"{min(range_mse_list):.6f}",
                f"{max(range_mse_list):.6f}",
                f"{sum(range_mae_list) / len(range_mae_list):.6f}",
                f"{min(range_mae_list):.6f}",
                f"{max(range_mae_list):.6f}"
            ])
    
    if range_table_data:
        print(tabulate(range_table_data, headers=range_table_headers, tablefmt="grid"))
    
    print("\n" + "="*80)
    print(f"Pred_Len={pred_len} 分析完成！")
    print("="*80)

def analyze_results(results_by_pred_len):
    """分析多个预测长度的结果"""
    # 定义每个预测长度的基准值
    baselines = {
        96: {'mse': None, 'mae': None},  # 96的基准值需要从实际结果中获取或手动设置
        720: {'mse': 0.462425, 'mae': 0.458175}  # 720的基准值（seed=2088）
    }
    
    # 过滤掉没有数据的预测长度
    pred_lens_with_data = [pred_len for pred_len in sorted(results_by_pred_len.keys()) 
                          if len(results_by_pred_len[pred_len]) > 0]
    pred_lens_without_data = [pred_len for pred_len in sorted(results_by_pred_len.keys()) 
                             if len(results_by_pred_len[pred_len]) == 0]
    
    # 显示没有数据的预测长度提示
    if pred_lens_without_data:
        print("="*80)
        print("⚠️  提示：以下预测长度没有找到实验结果")
        print("="*80)
        for pred_len in pred_lens_without_data:
            print(f"  - Pred_Len: {pred_len}")
        print("\n可能的原因：")
        print("  1. 该预测长度的训练尚未运行")
        print("  2. 训练脚本中未包含该预测长度")
        print("  3. model_id_prefix 不匹配")
        print("\n" + "="*80 + "\n")
    
    if not pred_lens_with_data:
        print("="*80)
        print("❌ 未找到任何实验结果")
        print("="*80)
        print("\n请检查：")
        print("  1. 训练脚本是否已运行")
        print("  2. model_id_prefix 是否正确")
        print("  3. 结果文件路径是否正确")
        return
    
    # 对每个有数据的预测长度分别进行分析
    for idx, pred_len in enumerate(pred_lens_with_data):
        results = results_by_pred_len[pred_len]
        baseline_mse = baselines.get(pred_len, {}).get('mse')
        baseline_mae = baselines.get(pred_len, {}).get('mae')
        
        analyze_results_for_pred_len(results, pred_len, baseline_mse, baseline_mae)
        
        # 在不同预测长度之间添加分隔
        if idx < len(pred_lens_with_data) - 1:
            print("\n\n" + "="*80)
            print("="*80)
            print("\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='分析最佳参数组合多种子训练结果')
    parser.add_argument('--result_file', type=str, default=None, help='结果文件路径（默认: experiment_results.log）')
    parser.add_argument('--model_id_prefix', type=str, default='T3Time_FreTS_Gated_Qwen_BestParams_MultiSeed',
                       help='模型ID前缀（优先匹配）')
    parser.add_argument('--model_name', type=str, default=None,
                       help='模型名称（如果model_id_prefix匹配不到，则使用model_name匹配，默认从model_id_prefix推断）')
    parser.add_argument('--pred_len', type=int, nargs='+', default=[96, 720], 
                       help='预测长度列表（默认: 96 720）')
    args = parser.parse_args()
    
    results_by_pred_len = load_multi_seed_results(args.result_file, args.model_id_prefix, args.model_name, args.pred_len)
    analyze_results(results_by_pred_len)

if __name__ == "__main__":
    main()
