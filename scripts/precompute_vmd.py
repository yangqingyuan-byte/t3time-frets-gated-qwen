#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预计算 VMD 分量并缓存到磁盘。

示例：
python scripts/precompute_vmd.py \
    --data_csv ./dataset/ETT-small/ETTh1.csv \
    --seq_len 96 \
    --stride 96 \
    --k_modes 4 \
    --out_dir ./vmd_cache/ETTh1_k4_s96
"""
import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd

from utils.vmd_processor import compute_vmd


def sliding_windows(arr: np.ndarray, seq_len: int, stride: int):
    """生成滑动窗口起始索引。"""
    start = 0
    L = arr.shape[0]
    while start + seq_len <= L:
        yield start
        start += stride


def main():
    parser = argparse.ArgumentParser(description="Precompute VMD components with caching.")
    parser.add_argument("--data_csv", type=str, required=True, help="输入CSV路径，例如 ./dataset/ETT-small/ETTh1.csv")
    parser.add_argument("--seq_len", type=int, default=96, help="序列长度")
    parser.add_argument("--stride", type=int, default=96, help="滑窗步长")
    parser.add_argument("--k_modes", type=int, default=4, help="VMD 模态数 K")
    parser.add_argument("--alpha", type=float, default=2000.0, help="VMD 带宽约束 alpha")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录，用于保存 .npy 窗口文件")
    parser.add_argument("--time_col", type=str, default=None, help="时间列名（如果提供则会丢弃）")
    parser.add_argument("--exclude_cols", type=str, nargs="*", default=None, help="需要排除的列名列表")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 读取数据
    df = pd.read_csv(args.data_csv)
    # 去除时间列
    if args.time_col and args.time_col in df.columns:
        df = df.drop(columns=[args.time_col])
    # 排除指定列
    if args.exclude_cols:
        for c in args.exclude_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

    data = df.values.astype(np.float32)  # [L, N]
    L, N = data.shape
    print(f"Loaded data: {data.shape}, saving to {args.out_dir}")

    meta = {
        "data_csv": args.data_csv,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "k_modes": args.k_modes,
        "alpha": args.alpha,
        "time_col": args.time_col,
        "exclude_cols": args.exclude_cols,
        "num_features": int(N),
        "total_length": int(L),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    count = 0
    for start in sliding_windows(data, args.seq_len, args.stride):
        window = data[start : start + args.seq_len]  # [seq_len, N]
        # 为每个特征做 VMD: 输出 shape [K, seq_len]，堆叠为 [seq_len, N, K]
        imfs_list: List[np.ndarray] = []
        for j in range(N):
            imfs = compute_vmd(window[:, j], K=args.k_modes, alpha=args.alpha)
            imfs_list.append(imfs)  # [K, seq_len]
        # stack: N 轴在中间 -> [N, K, seq_len] -> 转为 [seq_len, N, K]
        imfs_arr = np.stack(imfs_list, axis=0)               # [N, K, seq_len]
        imfs_arr = np.transpose(imfs_arr, (2, 0, 1))         # [seq_len, N, K]

        out_path = os.path.join(args.out_dir, f"window_{count:06d}.npy")
        np.save(out_path, imfs_arr.astype(np.float32))
        count += 1
        if count % 50 == 0:
            print(f"saved {count} windows...")

    print(f"Done. Total windows: {count}")


if __name__ == "__main__":
    main()

