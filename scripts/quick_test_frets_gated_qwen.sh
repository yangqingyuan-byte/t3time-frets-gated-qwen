#!/bin/bash

# 快速测试 T3Time_FreTS_Gated_Qwen 模型
# 数据集: ETTh1
# 预测长度: 96

export PYTHONPATH=$PYTHONPATH:$(pwd)

python train_frets_gated_qwen.py \
    --data_path ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --num_nodes 7 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --dropout_n 0.1 \
    --channel 32 \
    --epochs 2 \
    --es_patience 5 \
    --embed_version qwen3_0.6b \
    --seed 2024

