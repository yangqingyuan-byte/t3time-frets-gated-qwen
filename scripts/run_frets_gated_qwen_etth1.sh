#!/bin/bash

# 设置 PYTHONPATH 确保能找到项目模块
export PYTHONPATH=$PYTHONPATH:$(pwd)

# --- 固定参数 ---
DATA_PATH="ETTh1"
MODEL_NAME="T3Time_FreTS_Gated_Qwen"
SEQ_LEN=96
BATCH_SIZE=16
LR=1e-4
CHANNEL=64
EPOCHS=100
PATIENCE=10

# --- 预测长度 ---
PRED_LENS=(96)

echo "开始 $MODEL_NAME 在 $DATA_PATH 上的 2020-2070 种子序贯实验..."

for PRED_LEN in "${PRED_LENS[@]}"
do
    # 使用 seq 生成 2020 到 2070 的序列
    for SEED in $(seq 2020 2070)
    do
        echo "=================================================="
        echo "正在运行: Pred_Len=$PRED_LEN | Seed=$SEED"
        echo "=================================================="
        
        python train_frets_gated_qwen.py \
            --data_path $DATA_PATH \
            --seq_len $SEQ_LEN \
            --pred_len $PRED_LEN \
            --num_nodes 7 \
            --batch_size $BATCH_SIZE \
            --learning_rate $LR \
            --dropout_n 0.1 \
            --channel $CHANNEL \
            --epochs $EPOCHS \
            --es_patience $PATIENCE \
            --embed_version qwen3_0.6b \
            --seed $SEED
            
        echo "Seed $SEED 实验完成。"
    done
    echo "Pred_Len $PRED_LEN 所有种子 (2020-2070) 实验完成。"
done

echo "所有实验已完成！请使用您的筛选分析脚本查看结果。"
