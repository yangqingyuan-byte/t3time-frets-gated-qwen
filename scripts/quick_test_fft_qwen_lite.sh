#!/bin/bash
echo "🚀 快速测试 T3Time_FFT_Qwen_Lite (1 Epoch)"
python train_fft_qwen_lite.py \
    --data_path ETTh1 \
    --batch_size 16 \
    --epochs 1 \
    --channel 32 \
    --seed 2024

