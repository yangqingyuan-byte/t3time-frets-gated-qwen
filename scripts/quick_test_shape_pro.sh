#!/bin/bash
# 快速测试 Pro 版模型逻辑
python train_wavelet_gated_shape_pro_qwen.py \
    --data_path ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --channel 32 \
    --epochs 1 \
    --batch_size 4 \
    --seed 2031 \
    --embed_version qwen3_0.6b \
    --shape_lambda 0.1

if [ $? -eq 0 ]; then
    echo "✅ Pro 版模型快速测试通过！"
else
    echo "❌ Pro 版模型运行出错，请检查代码。"
fi
