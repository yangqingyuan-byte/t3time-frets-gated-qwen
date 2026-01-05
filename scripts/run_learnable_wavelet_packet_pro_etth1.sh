#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# 恢复平衡参数 + 开启门控融合
python train_learnable_wavelet_packet_pro.py \
    --model_id T3Time_Pro_Qwen_SOTA_V30 \
    --model TriModalLearnableWaveletPacketGatedProQwen \
    --data_path ETTh1 \
    --pred_len 96 \
    --channel 128 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --dropout_n 0.5 \
    --weight_decay 1e-3 \
    --wp_level 2 \
    --lradj type1 \
    --epochs 100 \
    --patience 15 \
    --seed 2024 \
    --gpu 0

echo "Pro Version Optimized Test Finished."
