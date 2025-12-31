#!/bin/bash

# 快速测试脚本：T3Time_FFT_Qwen
# 只运行 1 个 epoch，用于检测运行和日志记录是否正常

echo "=========================================="
echo "快速测试: T3Time_FFT_Qwen"
echo "=========================================="

python train_fft_qwen.py \
    --data_path ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --num_nodes 7 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --dropout_n 0.3 \
    --channel 32 \
    --epochs 1 \
    --es_patience 25 \
    --embed_version qwen3_0.6b \
    --seed 2024

echo "=========================================="
echo "测试完成！"
echo "请检查 experiment_results.log 是否记录了结果"
echo "=========================================="

