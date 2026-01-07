#!/usr/bin/env python3
"""
T3Time_FreEformer_Gated_Qwen 阶段2参数寻优结果分析脚本
分析 learning_rate, dropout_n, batch_size 的寻优结果
"""

import json
import argparse
from collections import defaultdict
from datetime import datetime

def load_stage2_results(result_file=None, model_id_prefix="T3Time_FreEformer_Stage2"):
    """
    从 experiment_results.log 加载阶段2的结果
    """
    if result_file is None:
        result_file = "/root/0/T3Time/experiment_results.log"
    
    results = []
    
    try:
        with open(result_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 检查是否是阶段2的结果
                    if model_id_prefix in data.get('model_id', ''):
                        results.append(data)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"❌ 结果文件不存在: {result_file}")
        return []
    
    return results

def analyze_step2_1_learning_rate(results):
    """分析步骤2.1: learning_rate 寻优结果"""
    step_results = [r for r in results if 'Step2_1' in r.get('model_id', '')]
    
    if not step_results:
        return None
    
    print("=" * 80)
    print("步骤 2.1: Learning_Rate 寻优结果")
    print("=" * 80)
    
    lr_results = {}
    for r in step_results:
        lr = r.get('learning_rate')
        mse = r.get('test_mse')
        mae = r.get('test_mae')
        if lr is not None and mse is not None:
            lr_results[lr] = {
                'mse': mse,
                'mae': mae,
                'timestamp': r.get('timestamp', '')
            }
    
    if not lr_results:
        print("❌ 未找到有效的 Learning_Rate 结果")
        return None
    
    sorted_lrs = sorted(lr_results.items(), key=lambda x: x[1]['mse'])
    
    print(f"\n找到 {len(sorted_lrs)} 个 Learning_Rate 配置的结果:\n")
    print(f"{'Learning_Rate':<15} {'MSE':<15} {'MAE':<15} {'Timestamp':<20}")
    print("-" * 80)
    
    for lr, metrics in sorted_lrs:
        print(f"{lr:<15.6e} {metrics['mse']:<15.6f} {metrics['mae']:<15.6f} {metrics['timestamp']:<20}")
    
    best_lr, best_metrics = sorted_lrs[0]
    print(f"\n🏆 最佳 Learning_Rate: {best_lr}")
    print(f"   MSE: {best_metrics['mse']:.6f}")
    print(f"   MAE: {best_metrics['mae']:.6f}")
    
    return best_lr

def float_to_scientific_str(value):
    """将浮点数转换为科学计数法字符串（用于匹配模型ID）"""
    if isinstance(value, str):
        return value
    # 转换为科学计数法
    if value >= 1e-3:
        return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    else:
        # 对于小于0.001的值，使用更精确的格式
        if abs(value - 1e-4) < 1e-6:
            return "1e-4"
        elif abs(value - 1.5e-4) < 1e-6:
            return "1.5e-4"
        elif abs(value - 7.5e-5) < 1e-6:
            return "7.5e-5"
        elif abs(value - 5e-5) < 1e-6:
            return "5e-5"
        else:
            return f"{value:.2e}".replace("e-0", "e-").replace("e+0", "e+")

def analyze_step2_2_dropout(results, best_learning_rate):
    """分析步骤2.2: dropout_n 寻优结果"""
    # 将learning_rate转换为模型ID中使用的格式
    lr_str = float_to_scientific_str(best_learning_rate)
    
    # 匹配模型ID（支持多种格式）
    step_results = []
    for r in results:
        model_id = r.get('model_id', '')
        if 'Step2_2' in model_id:
            # 检查是否包含对应的learning_rate（支持多种格式）
            if f'LR{lr_str}' in model_id or f'LR{best_learning_rate}' in model_id:
                step_results.append(r)
            # 也检查实际的learning_rate字段是否匹配
            elif abs(r.get('learning_rate', 0) - best_learning_rate) < 1e-8:
                step_results.append(r)
    
    if not step_results:
        # 如果没找到，尝试查找所有Step2_2的结果（可能是脚本使用了不同的learning_rate）
        all_step2_2 = [r for r in results if 'Step2_2' in r.get('model_id', '')]
        if all_step2_2:
            print(f"⚠️  警告: 未找到使用 Learning_Rate={best_learning_rate} ({lr_str}) 的步骤2.2结果")
            print(f"   但找到了 {len(all_step2_2)} 个步骤2.2的实验结果")
            # 显示实际使用的learning_rate
            actual_lrs = set()
            for r in all_step2_2:
                actual_lrs.add(r.get('learning_rate'))
            print(f"   实际使用的 Learning_Rate 值: {sorted(actual_lrs)}")
            # 使用实际找到的第一个learning_rate进行分析
            if actual_lrs:
                actual_lr = sorted(actual_lrs)[0]
                print(f"   将使用实际找到的 Learning_Rate: {actual_lr}")
                return analyze_step2_2_dropout(results, actual_lr)
        return None
    
    print("\n" + "=" * 80)
    print(f"步骤 2.2: Dropout 寻优结果（Learning_Rate={best_learning_rate}）")
    print("=" * 80)
    
    dropout_results = {}
    for r in step_results:
        dropout = r.get('dropout_n')
        mse = r.get('test_mse')
        mae = r.get('test_mae')
        if dropout is not None and mse is not None:
            dropout_results[dropout] = {
                'mse': mse,
                'mae': mae,
                'timestamp': r.get('timestamp', '')
            }
    
    if not dropout_results:
        print("❌ 未找到有效的 Dropout 结果")
        return None
    
    sorted_dropouts = sorted(dropout_results.items(), key=lambda x: x[1]['mse'])
    
    print(f"\n找到 {len(sorted_dropouts)} 个 Dropout 配置的结果:\n")
    print(f"{'Dropout':<15} {'MSE':<15} {'MAE':<15} {'Timestamp':<20}")
    print("-" * 80)
    
    for dropout, metrics in sorted_dropouts:
        print(f"{dropout:<15.3f} {metrics['mse']:<15.6f} {metrics['mae']:<15.6f} {metrics['timestamp']:<20}")
    
    best_dropout, best_metrics = sorted_dropouts[0]
    print(f"\n🏆 最佳 Dropout: {best_dropout}")
    print(f"   MSE: {best_metrics['mse']:.6f}")
    print(f"   MAE: {best_metrics['mae']:.6f}")
    
    return best_dropout

def analyze_step2_3_batch_size(results, best_learning_rate, best_dropout):
    """分析步骤2.3: batch_size 寻优结果"""
    # 将learning_rate转换为模型ID中使用的格式
    lr_str = float_to_scientific_str(best_learning_rate)
    
    # 匹配模型ID（支持多种格式）
    step_results = []
    for r in results:
        model_id = r.get('model_id', '')
        if 'Step2_3' in model_id:
            # 检查learning_rate和dropout是否匹配
            lr_match = f'LR{lr_str}' in model_id or f'LR{best_learning_rate}' in model_id or abs(r.get('learning_rate', 0) - best_learning_rate) < 1e-8
            dropout_match = f'Dropout{best_dropout}' in model_id or abs(r.get('dropout_n', -1) - best_dropout) < 1e-6
            
            if lr_match and dropout_match:
                step_results.append(r)
    
    if not step_results:
        # 如果没找到，尝试查找所有Step2_3的结果
        all_step2_3 = [r for r in results if 'Step2_3' in r.get('model_id', '')]
        if all_step2_3:
            print(f"⚠️  警告: 未找到使用 Learning_Rate={best_learning_rate}, Dropout={best_dropout} 的步骤2.3结果")
            print(f"   但找到了 {len(all_step2_3)} 个步骤2.3的实验结果")
            # 显示实际使用的参数
            actual_params = set()
            for r in all_step2_3:
                actual_params.add((r.get('learning_rate'), r.get('dropout_n')))
            print(f"   实际使用的参数组合: {sorted(actual_params)}")
            # 使用实际找到的第一个参数组合进行分析
            if actual_params:
                actual_lr, actual_dropout = sorted(actual_params)[0]
                print(f"   将使用实际找到的参数: Learning_Rate={actual_lr}, Dropout={actual_dropout}")
                return analyze_step2_3_batch_size(results, actual_lr, actual_dropout)
        return None
    
    print("\n" + "=" * 80)
    print(f"步骤 2.3: Batch_Size 寻优结果（Learning_Rate={best_learning_rate}, Dropout={best_dropout}）")
    print("=" * 80)
    
    batch_results = {}
    for r in step_results:
        batch_size = r.get('batch_size')
        mse = r.get('test_mse')
        mae = r.get('test_mae')
        if batch_size is not None and mse is not None:
            batch_results[batch_size] = {
                'mse': mse,
                'mae': mae,
                'timestamp': r.get('timestamp', '')
            }
    
    if not batch_results:
        print("❌ 未找到有效的 Batch_Size 结果")
        return None
    
    sorted_batches = sorted(batch_results.items(), key=lambda x: x[1]['mse'])
    
    print(f"\n找到 {len(sorted_batches)} 个 Batch_Size 配置的结果:\n")
    print(f"{'Batch_Size':<15} {'MSE':<15} {'MAE':<15} {'Timestamp':<20}")
    print("-" * 80)
    
    for batch_size, metrics in sorted_batches:
        print(f"{batch_size:<15} {metrics['mse']:<15.6f} {metrics['mae']:<15.6f} {metrics['timestamp']:<20}")
    
    best_batch, best_metrics = sorted_batches[0]
    print(f"\n🏆 最佳 Batch_Size: {best_batch}")
    print(f"   MSE: {best_metrics['mse']:.6f}")
    print(f"   MAE: {best_metrics['mae']:.6f}")
    
    return best_batch, best_metrics

def analyze_all_results(results):
    """综合分析所有阶段2的结果"""
    print("\n" + "=" * 80)
    print("阶段2 综合分析")
    print("=" * 80)
    
    # 步骤2.1: Learning_Rate
    best_learning_rate = analyze_step2_1_learning_rate(results)
    if best_learning_rate is None:
        print("\n❌ 无法继续分析，缺少步骤2.1的结果")
        return
    
    # 步骤2.2: Dropout（可能会自动调整learning_rate）
    best_dropout = analyze_step2_2_dropout(results, best_learning_rate)
    if best_dropout is None:
        print("\n❌ 无法继续分析，缺少步骤2.2的结果")
        return
    
    # 检查步骤2.2实际使用的learning_rate
    step2_2_results = [r for r in results if 'Step2_2' in r.get('model_id', '')]
    actual_lr_step2_2 = None
    if step2_2_results:
        actual_lr_step2_2 = step2_2_results[0].get('learning_rate')
        if abs(actual_lr_step2_2 - best_learning_rate) > 1e-8:
            print(f"\n⚠️  注意: 步骤2.2实际使用的 Learning_Rate={actual_lr_step2_2}，而不是步骤2.1的最佳值 {best_learning_rate}")
            best_learning_rate = actual_lr_step2_2
    
    # 步骤2.3: Batch_Size（可能会自动调整参数）
    batch_result = analyze_step2_3_batch_size(results, best_learning_rate, best_dropout)
    if batch_result is None:
        print("\n❌ 无法继续分析，缺少步骤2.3的结果")
        return
    
    best_batch_size, final_metrics = batch_result
    
    # 检查步骤2.3实际使用的参数
    step2_3_results = [r for r in results if 'Step2_3' in r.get('model_id', '')]
    actual_lr_step2_3 = None
    actual_dropout_step2_3 = None
    if step2_3_results:
        actual_lr_step2_3 = step2_3_results[0].get('learning_rate')
        actual_dropout_step2_3 = step2_3_results[0].get('dropout_n')
        if abs(actual_lr_step2_3 - best_learning_rate) > 1e-8:
            print(f"\n⚠️  注意: 步骤2.3实际使用的 Learning_Rate={actual_lr_step2_3}")
            best_learning_rate = actual_lr_step2_3
        if abs(actual_dropout_step2_3 - best_dropout) > 1e-6:
            print(f"\n⚠️  注意: 步骤2.3实际使用的 Dropout={actual_dropout_step2_3}")
            best_dropout = actual_dropout_step2_3
    
    # 总结
    print("\n" + "=" * 80)
    print("🎯 阶段2 最佳参数组合（实际使用的参数）")
    print("=" * 80)
    print(f"  Learning_Rate: {best_learning_rate}")
    print(f"  Dropout:        {best_dropout}")
    print(f"  Batch_Size:     {best_batch_size}")
    print(f"  最终 MSE:       {final_metrics['mse']:.6f}")
    print(f"  最终 MAE:       {final_metrics['mae']:.6f}")
    print("=" * 80)
    
    # 参数影响分析
    print("\n" + "=" * 80)
    print("参数影响分析")
    print("=" * 80)
    
    # Learning_Rate 影响
    lr_results = defaultdict(list)
    for r in results:
        if 'Step2_1' in r.get('model_id', ''):
            lr = r.get('learning_rate')
            mse = r.get('test_mse')
            if lr is not None and mse is not None:
                lr_results[lr].append(mse)
    
    if lr_results:
        print("\nLearning_Rate 参数影响:")
        for lr in sorted(lr_results.keys()):
            mses = lr_results[lr]
            avg_mse = sum(mses) / len(mses)
            print(f"  Learning_Rate {lr:.6e}: 平均 MSE = {avg_mse:.6f} (样本数: {len(mses)})")
    
    # Dropout 影响
    dropout_results = defaultdict(list)
    for r in results:
        if 'Step2_2' in r.get('model_id', ''):
            dropout = r.get('dropout_n')
            mse = r.get('test_mse')
            if dropout is not None and mse is not None:
                dropout_results[dropout].append(mse)
    
    if dropout_results:
        print("\nDropout 参数影响:")
        for dropout in sorted(dropout_results.keys()):
            mses = dropout_results[dropout]
            avg_mse = sum(mses) / len(mses)
            print(f"  Dropout {dropout:.1f}: 平均 MSE = {avg_mse:.6f} (样本数: {len(mses)})")
    
    # Batch_Size 影响
    batch_results = defaultdict(list)
    for r in results:
        if 'Step2_3' in r.get('model_id', ''):
            batch_size = r.get('batch_size')
            mse = r.get('test_mse')
            if batch_size is not None and mse is not None:
                batch_results[batch_size].append(mse)
    
    if batch_results:
        print("\nBatch_Size 参数影响:")
        for batch_size in sorted(batch_results.keys()):
            mses = batch_results[batch_size]
            avg_mse = sum(mses) / len(mses)
            print(f"  Batch_Size {batch_size}: 平均 MSE = {avg_mse:.6f} (样本数: {len(mses)})")
    
    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)
    print(f"\n阶段1和阶段2的完整最佳参数组合:")
    print(f"  --channel 32")
    print(f"  --fre_e_layer 1")
    print(f"  --embed_size 8")
    print(f"  --learning_rate {best_learning_rate}")
    print(f"  --dropout_n {best_dropout}")
    print(f"  --batch_size {best_batch_size}")

def main():
    parser = argparse.ArgumentParser(description='分析 T3Time_FreEformer_Gated_Qwen 阶段2参数寻优结果')
    parser.add_argument('--result_file', type=str, default=None,
                       help='结果文件路径（默认: /root/0/T3Time/experiment_results.log）')
    parser.add_argument('--model_id_prefix', type=str, default='T3Time_FreEformer_Stage2',
                       help='模型ID前缀（默认: T3Time_FreEformer_Stage2）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("T3Time_FreEformer_Gated_Qwen 阶段2参数寻优结果分析")
    print("=" * 80)
    
    results = load_stage2_results(args.result_file, args.model_id_prefix)
    
    if not results:
        print(f"\n❌ 未找到阶段2的实验结果（model_id_prefix: {args.model_id_prefix}）")
        print("\n请确保:")
        print("  1. 已运行阶段2寻优脚本: bash scripts/T3Time_FreEformer/hyperopt_stage2.sh")
        print("  2. 结果已保存到 experiment_results.log")
        return
    
    print(f"\n找到 {len(results)} 条阶段2实验结果")
    
    analyze_all_results(results)

if __name__ == "__main__":
    main()
