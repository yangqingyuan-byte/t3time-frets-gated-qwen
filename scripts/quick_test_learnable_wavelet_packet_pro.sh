#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# Pro 版快速测试脚本
python train_learnable_wavelet_packet_pro.py \
    --data_path ETTh1 \
    --pred_len 96 \
    --channel 32 \
    --batch_size 32 \
    --dropout_n 0.1 \
    --epochs 2 \
    --seed 2024

echo "Pro Version Quick Test Finished."
