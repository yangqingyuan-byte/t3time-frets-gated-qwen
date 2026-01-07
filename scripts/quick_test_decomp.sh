#!/bin/bash
echo "开始快速测试 T3Time_Wavelet_Decomp_Gated_Qwen..."

python train_wavelet_decomp_gated_qwen.py \
    --data_path ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --num_nodes 7 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --dropout_n 0.1 \
    --channel 32 \
    --epochs 1 \
    --seed 2026 \
    --embed_version qwen3_0.6b

if [ $? -eq 0 ]; then
    echo "✅ 测试成功！模型运行、日志记录均正常。"
else
    echo "❌ 测试失败！请检查报错信息。"
    exit 1
fi

