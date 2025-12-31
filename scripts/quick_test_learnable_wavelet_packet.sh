#!/bin/bash

# 设置显卡
export CUDA_VISIBLE_DEVICES=1

# 快速测试脚本：验证可学习小波包模型是否能正常跑通
python train_learnable_wavelet_packet_gated_qwen.py \
    --data_path ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout_n 0.1 \
    --channel 32 \
    --wp_level 2 \
    --epochs 2 \
    --patience 1 \
    --seed 2024 \
    --embed_version qwen3_0.6b

echo "Learnable Wavelet Packet Quick Test Finished."
