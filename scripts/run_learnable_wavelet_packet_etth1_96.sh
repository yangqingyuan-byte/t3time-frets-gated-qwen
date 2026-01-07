#!/bin/bash

# 设置显卡
export CUDA_VISIBLE_DEVICES=1

# 标准测试脚本：在 ETTh1 96 预测长度上进行完整训练和测试
# 使用推荐的稳定超参数
python train_learnable_wavelet_packet_gated_qwen.py \
    --data_path ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --channel 256 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --dropout_n 0.4 \
    --wp_level 2 \
    --epochs 100 \
    --patience 20 \
    --seed 2024 \
    --embed_version qwen3_0.6b

echo "------------------------------------------------------"
echo "ETTh1 pred_len=96 正常测试完成。"
