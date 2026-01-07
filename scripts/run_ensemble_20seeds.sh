#!/bin/bash

# 激活环境
source /root/anaconda3/bin/activate TimeCMA_Qwen3

# 生成20个种子：从2024到2043
SEEDS=(2024 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040 2041 2042 2043)

echo "========================================"
echo "Starting Ensemble Training with 20 Seeds"
echo "Total models to train: ${#SEEDS[@]}"
echo "========================================"

count=0
for seed in "${SEEDS[@]}"; do
    count=$((count + 1))
    echo "------------------------------------------------"
    echo "Training Model $count / ${#SEEDS[@]} - Seed: $seed"
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
echo "All 20 models finished training!"
echo "Now running ensemble evaluation..."
echo "========================================"

# 使用新的参数格式运行集成评估
python scripts/eval_ensemble.py --start_seed 2024 --num_seeds 20

echo "========================================"
echo "Ensemble evaluation completed!"
echo "========================================"
