#!/bin/bash
# 快速测试可学习小波版模型逻辑
export CUDA_VISIBLE_DEVICES=1
python train_learnable_wavelet_gated_shape_qwen.py \
    --data_path ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --channel 32 \
    --epochs 1 \
    --batch_size 4 \
    --seed 2031 \
    --embed_version qwen3_0.6b \
    --shape_lambda 0.1 \
    --levels 3

if [ $? -eq 0 ]; then
    echo "✅ 可学习小波版模型快速测试通过！"
else
    echo "❌ 可学习小波版模型运行出错，请检查代码。"
fi

