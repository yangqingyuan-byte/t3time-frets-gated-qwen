#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VMD 分支使用示例（演示前向通过），需要：
1) 预先生成的 VMD 窗口文件 (.npy)，形状 [seq_len, N, K]
2) TriModalVMD 模型

示例：
python scripts/run_vmd_example.py \
    --vmd_window ./vmd_cache/ETTh1_k4_s96/window_000000.npy \
    --channel 64 --head 8 --pred_len 96
"""
import argparse
import numpy as np
import torch

from models.T3Time_VMD import TriModalVMD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vmd_window", type=str, required=True, help="预先计算好的 VMD 窗口 .npy 文件路径")
    parser.add_argument("--channel", type=int, default=64)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device

    # 加载 VMD 窗口：形状 [L, N, K]
    vmd_arr = np.load(args.vmd_window)  # [L, N, K]
    L, N, K = vmd_arr.shape

    # 构造伪输入：batch=1
    input_data = torch.randn(1, L, N, device=device)          # [B, L, N]
    input_mark = torch.zeros(1, L, 1, device=device)          # 时间标记占位
    embeddings = torch.zeros(1, 1, N, device=device)          # 预训练嵌入占位
    x_modes = torch.tensor(vmd_arr, device=device).unsqueeze(0)  # [1, L, N, K]

    model = TriModalVMD(
        device=device,
        channel=args.channel,
        num_nodes=N,
        seq_len=L,
        pred_len=args.pred_len,
        head=args.head,
    ).to(device)

    with torch.no_grad():
        out = model(input_data, input_mark, embeddings, x_modes)
    print(f"Input: input_data {tuple(input_data.shape)}, x_modes {tuple(x_modes.shape)}")
    print(f"Output: {tuple(out.shape)}  # [B, pred_len, C]")


if __name__ == "__main__":
    main()

