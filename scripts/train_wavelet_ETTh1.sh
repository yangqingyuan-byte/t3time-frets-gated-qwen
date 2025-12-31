#!/bin/bash
# 训练基于小波变换的T3Time模型 - ETTh1数据集

export CUDA_VISIBLE_DEVICES=0

python train_wavelet.py \
    --data_path ETTh1 \
    --device cuda \
    --seq_len 96 \
    --pred_len 96 \
    --channel 256 \
    --num_nodes 7 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --dropout_n 0.4 \
    --e_layer 1 \
    --d_layer 1 \
    --head 8 \
    --epochs 150 \
    --wavelet db4 \
    --use_cross_attention \
    --seed 2024 \
    --save ./logs_wavelet/

