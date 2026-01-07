#!/bin/bash

# 激活环境
source /root/anaconda3/bin/activate TimeCMA_Qwen3

# 种子列表
SEEDS=(2024 2025 2026)

for seed in "${SEEDS[@]}"; do
    echo "------------------------------------------------"
    echo "Starting training with seed: $seed"
    echo "------------------------------------------------"
    
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
        --seed $seed \
        --gpu 0
        
    echo "Training with seed $seed completed."
    echo ""
done

echo "========================================"
echo "All training finished. Now running ensemble evaluation..."
echo "========================================"
python scripts/eval_ensemble.py
