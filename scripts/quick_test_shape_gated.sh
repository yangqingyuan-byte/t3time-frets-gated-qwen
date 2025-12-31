#!/bin/bash

# 快速测试脚本：验证带形态损失的模型运行和日志记录

echo "开始快速测试 T3Time_Wavelet_Gated_Shape_Qwen (Shape Loss)..."

python train_wavelet_gated_shape_qwen.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --num_nodes 7 \
  --batch_size 16 \
  --epochs 1 \
  --es_patience 1 \
  --embed_version qwen3_0.6b \
  --shape_lambda 0.1 \
  --seed 2026

if [ $? -eq 0 ]; then
    echo -e "\n✅ 测试成功！形态损失计算和日志记录正常。"
    tail -n 1 experiment_results.log
else
    echo -e "\n❌ 测试失败！"
fi

