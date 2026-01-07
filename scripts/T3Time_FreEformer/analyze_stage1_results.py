#!/usr/bin/env python3
"""
T3Time_FreEformer_Gated_Qwen 阶段1参数寻优结果分析脚本
分析 channel, fre_e_layer, embed_size 的寻优结果
"""

import json
import argparse
from collections import defaultdict
from datetime import datetime

def load_stage1_results(result_file=None, model_id_prefix="T3Time_FreEformer_Stage1"):
    """
    从 experiment_results.log 加载阶段1的结果
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
                    # 检查是否是阶段1的结果
                    if model_id_prefix in data.get('model_id', ''):
                        results.append(data)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"❌ 结果文件不存在: {result_file}")
        return []
    
    return results

def analyze_step1_1_channel(results):
    """分析步骤1.1: channel 寻优结果"""
    step_results = [r for r in results if 'Step1_1' in r.get('model_id', '')]
    
    if not step_results:
        return None
    
    print("=" * 80)
    print("步骤 1.1: Channel 寻优结果")
    print("=" * 80)
    
    channel_results = {}
    for r in step_results:
        channel = r.get('channel')
        mse = r.get('test_mse')
        mae = r.get('test_mae')
        if channel is not None and mse is not None:
            channel_results[channel] = {
                'mse': mse,
                'mae': mae,
                'timestamp': r.get('timestamp', '')
            }
    
    if not channel_results:
        print("❌ 未找到有效的 Channel 结果")
        return None
    
    # 按 MSE 排序
    sorted_channels = sorted(channel_results.items(), key=lambda x: x[1]['mse'])
    
    print(f"\n找到 {len(sorted_channels)} 个 Channel 配置的结果:\n")
    print(f"{'Channel':<10} {'MSE':<15} {'MAE':<15} {'Timestamp':<20}")
    print("-" * 80)
    
    for channel, metrics in sorted_channels:
        print(f"{channel:<10} {metrics['mse']:<15.6f} {metrics['mae']:<15.6f} {metrics['timestamp']:<20}")
    
    best_channel, best_metrics = sorted_channels[0]
    print(f"\n🏆 最佳 Channel: {best_channel}")
    print(f"   MSE: {best_metrics['mse']:.6f}")
    print(f"   MAE: {best_metrics['mae']:.6f}")
    
    return best_channel

def analyze_step1_2_fre_e_layer(results, best_channel):
    """分析步骤1.2: fre_e_layer 寻优结果"""
    step_results = [r for r in results if 'Step1_2' in r.get('model_id', '') and 
                    f'Channel{best_channel}' in r.get('model_id', '')]
    
    if not step_results:
        return None
    
    print("\n" + "=" * 80)
    print(f"步骤 1.2: Fre_E_Layer 寻优结果（Channel={best_channel}）")
    print("=" * 80)
    
    layer_results = {}
    for r in step_results:
        fre_e_layer = r.get('fre_e_layer')
        mse = r.get('test_mse')
        mae = r.get('test_mae')
        if fre_e_layer is not None and mse is not None:
            layer_results[fre_e_layer] = {
                'mse': mse,
                'mae': mae,
                'timestamp': r.get('timestamp', '')
            }
    
    if not layer_results:
        print("❌ 未找到有效的 Fre_E_Layer 结果")
        return None
    
    sorted_layers = sorted(layer_results.items(), key=lambda x: x[1]['mse'])
    
    print(f"\n找到 {len(sorted_layers)} 个 Fre_E_Layer 配置的结果:\n")
    print(f"{'Fre_E_Layer':<15} {'MSE':<15} {'MAE':<15} {'Timestamp':<20}")
    print("-" * 80)
    
    for layer, metrics in sorted_layers:
        print(f"{layer:<15} {metrics['mse']:<15.6f} {metrics['mae']:<15.6f} {metrics['timestamp']:<20}")
    
    best_layer, best_metrics = sorted_layers[0]
    print(f"\n🏆 最佳 Fre_E_Layer: {best_layer}")
    print(f"   MSE: {best_metrics['mse']:.6f}")
    print(f"   MAE: {best_metrics['mae']:.6f}")
    
    return best_layer

def analyze_step1_3_embed_size(results, best_channel, best_fre_e_layer):
    """分析步骤1.3: embed_size 寻优结果"""
    # 首先尝试匹配最佳 fre_e_layer
    step_results = [r for r in results if 'Step1_3' in r.get('model_id', '') and 
                    f'Channel{best_channel}' in r.get('model_id', '') and
                    f'FreELayer{best_fre_e_layer}' in r.get('model_id', '')]
    
    # 如果没有找到，尝试匹配所有 Step1_3 的结果（可能使用了不同的 fre_e_layer）
    if not step_results:
        step_results = [r for r in results if 'Step1_3' in r.get('model_id', '') and 
                        f'Channel{best_channel}' in r.get('model_id', '')]
    
    if not step_results:
        return None
    
    # 检查实际使用的 fre_e_layer
    actual_fre_e_layers = set(r.get('fre_e_layer') for r in step_results if r.get('fre_e_layer') is not None)
    
    print("\n" + "=" * 80)
    if len(actual_fre_e_layers) == 1 and list(actual_fre_e_layers)[0] != best_fre_e_layer:
        print(f"步骤 1.3: Embed_Size 寻优结果（Channel={best_channel}）")
        print(f"⚠️  注意: 实验实际使用的 Fre_E_Layer={list(actual_fre_e_layers)[0]}，而不是步骤1.2的最佳值 {best_fre_e_layer}")
    else:
        print(f"步骤 1.3: Embed_Size 寻优结果（Channel={best_channel}, Fre_E_Layer={best_fre_e_layer}）")
    print("=" * 80)
    
    embed_results = {}
    for r in step_results:
        embed_size = r.get('embed_size')
        mse = r.get('test_mse')
        mae = r.get('test_mae')
        if embed_size is not None and mse is not None:
            embed_results[embed_size] = {
                'mse': mse,
                'mae': mae,
                'timestamp': r.get('timestamp', '')
            }
    
    if not embed_results:
        print("❌ 未找到有效的 Embed_Size 结果")
        return None
    
    sorted_embeds = sorted(embed_results.items(), key=lambda x: x[1]['mse'])
    
    print(f"\n找到 {len(sorted_embeds)} 个 Embed_Size 配置的结果:\n")
    print(f"{'Embed_Size':<15} {'MSE':<15} {'MAE':<15} {'Timestamp':<20}")
    print("-" * 80)
    
    for embed, metrics in sorted_embeds:
        print(f"{embed:<15} {metrics['mse']:<15.6f} {metrics['mae']:<15.6f} {metrics['timestamp']:<20}")
    
    best_embed, best_metrics = sorted_embeds[0]
    print(f"\n🏆 最佳 Embed_Size: {best_embed}")
    print(f"   MSE: {best_metrics['mse']:.6f}")
    print(f"   MAE: {best_metrics['mae']:.6f}")
    
    return best_embed, best_metrics

def analyze_all_results(results):
    """综合分析所有阶段1的结果"""
    print("\n" + "=" * 80)
    print("阶段1 综合分析")
    print("=" * 80)
    
    # 步骤1.1: Channel
    best_channel = analyze_step1_1_channel(results)
    if best_channel is None:
        print("\n❌ 无法继续分析，缺少步骤1.1的结果")
        return
    
    # 步骤1.2: Fre_E_Layer
    best_fre_e_layer = analyze_step1_2_fre_e_layer(results, best_channel)
    if best_fre_e_layer is None:
        print("\n❌ 无法继续分析，缺少步骤1.2的结果")
        return
    
    # 步骤1.3: Embed_Size
    embed_result = analyze_step1_3_embed_size(results, best_channel, best_fre_e_layer)
    if embed_result is None:
        print("\n❌ 无法继续分析，缺少步骤1.3的结果")
        return
    
    best_embed_size, final_metrics = embed_result
    
    # 总结
    print("\n" + "=" * 80)
    print("🎯 阶段1 最佳参数组合")
    print("=" * 80)
    print(f"  Channel:      {best_channel}")
    print(f"  Fre_E_Layer:  {best_fre_e_layer}")
    print(f"  Embed_Size:   {best_embed_size}")
    print(f"  最终 MSE:     {final_metrics['mse']:.6f}")
    print(f"  最终 MAE:     {final_metrics['mae']:.6f}")
    print("=" * 80)
    
    # 参数对比分析
    print("\n" + "=" * 80)
    print("参数影响分析")
    print("=" * 80)
    
    # Channel 影响
    channel_results = defaultdict(list)
    for r in results:
        if 'Step1_1' in r.get('model_id', ''):
            channel = r.get('channel')
            mse = r.get('test_mse')
            if channel is not None and mse is not None:
                channel_results[channel].append(mse)
    
    if channel_results:
        print("\nChannel 参数影响:")
        for channel in sorted(channel_results.keys()):
            mses = channel_results[channel]
            avg_mse = sum(mses) / len(mses)
            print(f"  Channel {channel}: 平均 MSE = {avg_mse:.6f} (样本数: {len(mses)})")
    
    # Fre_E_Layer 影响
    layer_results = defaultdict(list)
    for r in results:
        if 'Step1_2' in r.get('model_id', ''):
            layer = r.get('fre_e_layer')
            mse = r.get('test_mse')
            if layer is not None and mse is not None:
                layer_results[layer].append(mse)
    
    if layer_results:
        print("\nFre_E_Layer 参数影响:")
        for layer in sorted(layer_results.keys()):
            mses = layer_results[layer]
            avg_mse = sum(mses) / len(mses)
            print(f"  Fre_E_Layer {layer}: 平均 MSE = {avg_mse:.6f} (样本数: {len(mses)})")
    
    # Embed_Size 影响
    embed_results = defaultdict(list)
    for r in results:
        if 'Step1_3' in r.get('model_id', ''):
            embed = r.get('embed_size')
            mse = r.get('test_mse')
            if embed is not None and mse is not None:
                embed_results[embed].append(mse)
    
    if embed_results:
        print("\nEmbed_Size 参数影响:")
        for embed in sorted(embed_results.keys()):
            mses = embed_results[embed]
            avg_mse = sum(mses) / len(mses)
            print(f"  Embed_Size {embed}: 平均 MSE = {avg_mse:.6f} (样本数: {len(mses)})")
    
    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)
    print(f"\n建议在阶段2中使用以下参数:")
    print(f"  --channel {best_channel}")
    print(f"  --fre_e_layer {best_fre_e_layer}")
    print(f"  --embed_size {best_embed_size}")

def main():
    parser = argparse.ArgumentParser(description='分析 T3Time_FreEformer_Gated_Qwen 阶段1参数寻优结果')
    parser.add_argument('--result_file', type=str, default=None,
                       help='结果文件路径（默认: /root/0/T3Time/experiment_results.log）')
    parser.add_argument('--model_id_prefix', type=str, default='T3Time_FreEformer_Stage1',
                       help='模型ID前缀（默认: T3Time_FreEformer_Stage1）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("T3Time_FreEformer_Gated_Qwen 阶段1参数寻优结果分析")
    print("=" * 80)
    
    results = load_stage1_results(args.result_file, args.model_id_prefix)
    
    if not results:
        print(f"\n❌ 未找到阶段1的实验结果（model_id_prefix: {args.model_id_prefix}）")
        print("\n请确保:")
        print("  1. 已运行阶段1寻优脚本: bash scripts/T3Time_FreEformer/hyperopt_stage1.sh")
        print("  2. 结果已保存到 experiment_results.log")
        return
    
    print(f"\n找到 {len(results)} 条阶段1实验结果")
    
    analyze_all_results(results)

if __name__ == "__main__":
    main()
