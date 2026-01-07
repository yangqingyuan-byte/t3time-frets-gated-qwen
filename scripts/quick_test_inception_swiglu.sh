#!/bin/bash

# 快速测试脚本：验证带 Inception 和 SwiGLU 的模型

echo "开始快速测试 T3Time_Inception_SwiGLU_Gated_Qwen..."

python train_inception_swiglu_gated_qwen.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --num_nodes 7 \
  --batch_size 16 \
  --epochs 1 \
  --es_patience 1 \
  --embed_version qwen3_0.6b \
  --seed 2026

if [ $? -eq 0 ]; then
    echo -e "\n✅ 测试成功！Inception 卷积、SwiGLU FFN 和日志记录均正常。"
    tail -n 1 experiment_results.log
else
    echo -e "\n❌ 测试失败！"
fi

