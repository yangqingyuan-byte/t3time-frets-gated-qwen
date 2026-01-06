#!/usr/bin/env python3
"""
分析 T3Time_FreEformer_Gated_Qwen 大范围参数寻优结果脚本
仿照 FreTS 的分析风格，从 experiment_results.log 中找出最优组合
"""

import json
import argparse
from collections import defaultdict


def load_results(result_file, model_id_prefix):
    results = []
    try:
        with open(result_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("model_id", "").startswith(model_id_prefix):
                    results.append(data)
    except FileNotFoundError:
        print(f"❌ 结果文件不存在: {result_file}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="分析 T3Time_FreEformer_Gated_Qwen 大范围参数寻优结果"
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="/root/0/T3Time/experiment_results.log",
        help="结果文件路径",
    )
    parser.add_argument(
        "--model_id_prefix",
        type=str,
        default="T3Time_FreEformer_Wide_ETTh1_pred96",
        help="模型ID前缀（与寻优脚本中的 MODEL_ID_PREFIX 对齐）",
    )
    parser.add_argument(
        "--topk", type=int, default=10, help="显示前多少个最优结果"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("T3Time_FreEformer_Gated_Qwen 大范围参数寻优结果分析")
    print("=" * 80)

    results = load_results(args.result_file, args.model_id_prefix)
    if not results:
        print(
            f"❌ 未找到以 '{args.model_id_prefix}' 开头的实验结果，请确认寻优脚本已运行完成。"
        )
        return

    print(f"共找到 {len(results)} 条实验结果")

    # 按 MSE 排序
    sorted_by_mse = sorted(results, key=lambda x: x.get("test_mse", 1e9))
    # 按 MAE 排序
    sorted_by_mae = sorted(results, key=lambda x: x.get("test_mae", 1e9))

    print("\n" + "=" * 80)
    print(f"🏆 Top {min(args.topk, len(sorted_by_mse))} 最优结果（按 Test MSE 排序）")
    print("=" * 80)
    print(
        f"{'Rank':<5} {'MSE':<10} {'MAE':<10} {'Channel':<8} "
        f"{'FreL':<5} {'Emb':<5} {'LR':<10} {'Drop':<6} "
        f"{'WD':<10} {'BS':<5} {'Seed':<6} {'Time':<19}"
    )
    print("-" * 80)

    for idx, r in enumerate(sorted_by_mse[: args.topk], start=1):
        print(
            f"{idx:<5}"
            f"{r.get('test_mse', 0):<10.6f}"
            f"{r.get('test_mae', 0):<10.6f}"
            f"{str(r.get('channel', '')):<8}"
            f"{str(r.get('fre_e_layer', '')):<5}"
            f"{str(r.get('embed_size', '')):<5}"
            f"{r.get('learning_rate', 0):<10.6f}"
            f"{r.get('dropout_n', 0):<6.2f}"
            f"{r.get('weight_decay', 0):<10.2e}"
            f"{str(r.get('batch_size', '')):<5}"
            f"{str(r.get('seed', '')):<6}"
            f"{str(r.get('timestamp', '')):<19}"
        )

    print("\n" + "=" * 80)
    print(f"🏆 Top {min(args.topk, len(sorted_by_mae))} 最优结果（按 Test MAE 排序）")
    print("=" * 80)
    print(
        f"{'Rank':<5} {'MSE':<10} {'MAE':<10} {'Channel':<8} "
        f"{'FreL':<5} {'Emb':<5} {'LR':<10} {'Drop':<6} "
        f"{'WD':<10} {'BS':<5} {'Seed':<6} {'Time':<19}"
    )
    print("-" * 80)

    for idx, r in enumerate(sorted_by_mae[: args.topk], start=1):
        print(
            f"{idx:<5}"
            f"{r.get('test_mse', 0):<10.6f}"
            f"{r.get('test_mae', 0):<10.6f}"
            f"{str(r.get('channel', '')):<8}"
            f"{str(r.get('fre_e_layer', '')):<5}"
            f"{str(r.get('embed_size', '')):<5}"
            f"{r.get('learning_rate', 0):<10.6f}"
            f"{r.get('dropout_n', 0):<6.2f}"
            f"{r.get('weight_decay', 0):<10.2e}"
            f"{str(r.get('batch_size', '')):<5}"
            f"{str(r.get('seed', '')):<6}"
            f"{str(r.get('timestamp', '')):<19}"
        )

    # 找出最佳 MSE
    best_mse = sorted_by_mse[0]
    # 找出最佳 MAE
    best_mae = sorted_by_mae[0]

    print("\n" + "=" * 80)
    print("🎯 最佳 MSE 参数组合")
    print("=" * 80)
    print(f"  Test MSE: {best_mse.get('test_mse', 0):.6f}")
    print(f"  Test MAE: {best_mse.get('test_mae', 0):.6f}")
    print(f"  Channel: {best_mse.get('channel')}")
    print(f"  Fre_E_Layer: {best_mse.get('fre_e_layer')}")
    print(f"  Embed_Size: {best_mse.get('embed_size')}")
    print(f"  Learning_Rate: {best_mse.get('learning_rate')}")
    print(f"  Dropout: {best_mse.get('dropout_n')}")
    print(f"  Weight_Decay: {best_mse.get('weight_decay')}")
    print(f"  Batch_Size: {best_mse.get('batch_size')}")
    print(f"  Seed: {best_mse.get('seed')}")
    print(f"  Timestamp: {best_mse.get('timestamp')}")

    print("\n" + "=" * 80)
    print("🎯 最佳 MAE 参数组合")
    print("=" * 80)
    print(f"  Test MSE: {best_mae.get('test_mse', 0):.6f}")
    print(f"  Test MAE: {best_mae.get('test_mae', 0):.6f}")
    print(f"  Channel: {best_mae.get('channel')}")
    print(f"  Fre_E_Layer: {best_mae.get('fre_e_layer')}")
    print(f"  Embed_Size: {best_mae.get('embed_size')}")
    print(f"  Learning_Rate: {best_mae.get('learning_rate')}")
    print(f"  Dropout: {best_mae.get('dropout_n')}")
    print(f"  Weight_Decay: {best_mae.get('weight_decay')}")
    print(f"  Batch_Size: {best_mae.get('batch_size')}")
    print(f"  Seed: {best_mae.get('seed')}")
    print(f"  Timestamp: {best_mae.get('timestamp')}")

    # 如果最佳MSE和最佳MAE是同一个结果，给出提示
    if best_mse.get('model_id') == best_mae.get('model_id'):
        print("\n" + "=" * 80)
        print("✅ 最佳 MSE 和最佳 MAE 是同一个实验！")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("ℹ️  最佳 MSE 和最佳 MAE 来自不同的实验")
        print("=" * 80)
        print(f"  最佳 MSE 的 MAE: {best_mse.get('test_mae', 0):.6f}")
        print(f"  最佳 MAE 的 MSE: {best_mae.get('test_mse', 0):.6f}")

    # 按 Channel 聚合的最优结果（按 MSE）
    print("\n" + "=" * 80)
    print("按 Channel 聚合的最优结果（按 MSE）")
    print("=" * 80)
    best_by_channel_mse = {}
    for r in results:
        ch = r.get("channel")
        mse = r.get("test_mse", 1e9)
        if ch not in best_by_channel_mse or mse < best_by_channel_mse[ch]["test_mse"]:
            best_by_channel_mse[ch] = r
    for ch in sorted(best_by_channel_mse.keys()):
        r = best_by_channel_mse[ch]
        print(
            f"  Channel {ch}: "
            f"MSE={r.get('test_mse', 0):.6f}, "
            f"MAE={r.get('test_mae', 0):.6f}, "
            f"Fre_E_Layer={r.get('fre_e_layer')}, "
            f"Embed_Size={r.get('embed_size')}, "
            f"LR={r.get('learning_rate')}, "
            f"Dropout={r.get('dropout_n')}, "
            f"BS={r.get('batch_size')}"
        )

    # 按 Channel 聚合的最优结果（按 MAE）
    print("\n" + "=" * 80)
    print("按 Channel 聚合的最优结果（按 MAE）")
    print("=" * 80)
    best_by_channel_mae = {}
    for r in results:
        ch = r.get("channel")
        mae = r.get("test_mae", 1e9)
        if ch not in best_by_channel_mae or mae < best_by_channel_mae[ch]["test_mae"]:
            best_by_channel_mae[ch] = r
    for ch in sorted(best_by_channel_mae.keys()):
        r = best_by_channel_mae[ch]
        print(
            f"  Channel {ch}: "
            f"MSE={r.get('test_mse', 0):.6f}, "
            f"MAE={r.get('test_mae', 0):.6f}, "
            f"Fre_E_Layer={r.get('fre_e_layer')}, "
            f"Embed_Size={r.get('embed_size')}, "
            f"LR={r.get('learning_rate')}, "
            f"Dropout={r.get('dropout_n')}, "
            f"BS={r.get('batch_size')}"
        )

    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

