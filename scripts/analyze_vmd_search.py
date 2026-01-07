#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据 VMD 超参搜索 CSV，统计每组超参对应实验的 Test MSE / MAE。

使用示例：
    python scripts/analyze_vmd_search.py \
        --csv logs_vmd/vmd_hparam_search_20251216-194701.csv \
        --output logs_vmd/vmd_hparam_search_summary.csv

约定：
    - 每个 log_dir 下有一个训练输出日志文件，例如 train.log
    - 日志中包含形如：
        Test MSE: 0.6472, Test MAE: 0.6123
      的行（train_vmd.py 的 stdout 重定向后即可）
"""

import os
import re
import csv
import argparse
from typing import Optional, Tuple


def parse_log_for_metrics(log_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    从指定日志文件中解析最后一次出现的 Test MSE / MAE。

    Args:
        log_path: 日志文件路径

    Returns:
        (mse, mae): 若未找到则返回 (None, None)
    """
    if not os.path.exists(log_path):
        return None, None

    mse = None
    mae = None

    pattern = re.compile(r"Test MSE:\s*([\d.]+),\s*Test MAE:\s*([\d.]+)")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in reversed(lines):
        m = pattern.search(line)
        if m:
            mse = float(m.group(1))
            mae = float(m.group(2))
            break

    return mse, mae


def analyze_csv(csv_path: str, output_path: str, log_filename: str = "train.log"):
    """
    读取超参搜索 CSV，针对每行的 log_dir 解析对应日志文件，汇总 Test MSE / MAE。
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("vmd_k"):
                continue
            rows.append(row)

    print(f"Loaded {len(rows)} hyperparam combos from {csv_path}")

    summary = []
    for row in rows:
        vmd_k = row["vmd_k"]
        vmd_alpha = row["vmd_alpha"]
        channel = row["channel"]
        log_dir = row["log_dir"]

        log_path = os.path.join(log_dir, log_filename)
        mse, mae = parse_log_for_metrics(log_path)

        status = "ok" if mse is not None else "missing_or_unparsed"
        print(
            f"k={vmd_k}, alpha={vmd_alpha}, c={channel} -> "
            f"MSE={mse}, MAE={mae}, status={status}"
        )

        summary.append(
            {
                "vmd_k": vmd_k,
                "vmd_alpha": vmd_alpha,
                "channel": channel,
                "log_dir": log_dir,
                "log_file": log_path,
                "test_mse": mse,
                "test_mae": mae,
                "status": status,
            }
        )

    # 写出汇总 CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "vmd_k",
                "vmd_alpha",
                "channel",
                "log_dir",
                "log_file",
                "test_mse",
                "test_mae",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nSummary written to: {output_path}")

    # 简单打印按 MSE 排序的前几名（只看 status=ok 的）
    valid_results = [r for r in summary if r["status"] == "ok" and r["test_mse"] is not None]
    valid_results.sort(key=lambda r: r["test_mse"])

    if valid_results:
        print("\nTop results by Test MSE:")
        for r in valid_results[:5]:
            print(
                f"k={r['vmd_k']}, alpha={r['vmd_alpha']}, c={r['channel']}, "
                f"MSE={r['test_mse']:.4f}, MAE={r['test_mae']:.4f}, dir={r['log_dir']}"
            )
    else:
        print("\nNo valid Test MSE/MAE found. "
              "Ensure you have saved stdout to log files under each log_dir.")


def parse_args():
    parser = argparse.ArgumentParser(description="分析 VMD 超参搜索结果")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="超参搜索结果 CSV 路径（如 logs_vmd/vmd_hparam_search_*.csv）",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出汇总 CSV 路径",
    )
    parser.add_argument(
        "--log_filename",
        type=str,
        default="train.log",
        help="每个 log_dir 中训练日志文件的文件名",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_csv(args.csv, args.output, args.log_filename)


