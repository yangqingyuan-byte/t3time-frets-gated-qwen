#!/bin/bash

# 设置 PYTHONPATH 确保能找到所有模块
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/plug_and_play

# 快速验证脚本：1个 epoch，较小的 batch_size，验证模型和 FreDFLoss 是否能正常跑通
python train_frets_gated_qwen_fredf.py \
    --data_path ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --num_nodes 7 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --epochs 1 \
    --es_patience 3 \
    --seed 2024 \
    --lambda_freq 0.5

