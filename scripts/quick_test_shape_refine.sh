#!/bin/bash
echo "开始快速测试 T3Time_Wavelet_Gated_Shape_Refine_Qwen..."

python train_wavelet_gated_shape_refine_qwen.py \
    --data_path ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --channel 32 \
    --epochs 1 \
    --seed 2031 \
    --batch_size 4 \
    --embed_version qwen3_0.6b

if [ $? -eq 0 ]; then
    echo "✅ 测试成功！Refine 版模型运行正常。"
else
    echo "❌ 测试失败！"
    exit 1
fi

