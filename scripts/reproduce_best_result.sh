#!/bin/bash
# 复现最佳结果（MSE=0.377742）的配置
# 根据分析，最佳结果使用的是：
# - channel=64, dropout=0.1, weight_decay=1e-4
# - scale=0.02 (默认值，当时未指定)
# - sparsity_threshold=0.01 (默认值，当时未指定)
# - loss_fn=smooth_l1 (默认值，当时未指定)
# - affine=True (默认值，当时未指定)

set -euo pipefail

export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "复现最佳结果配置"
echo "=========================================="
echo "目标: MSE=0.377742"
echo "配置: channel=64, dropout=0.1, weight_decay=1e-4"
echo "      scale=0.02 (默认), sparsity_threshold=0.01 (默认)"
echo "      loss_fn=smooth_l1 (默认), affine=True (需要检查)"
echo "=========================================="
echo ""

python train_frets_gated_qwen_fusion_exp.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --channel 64 \
  --dropout_n 0.1 \
  --weight_decay 1e-4 \
  --fusion_mode gate \
  --loss_fn smooth_l1 \
  --lradj type1 \
  --sparsity_threshold 0.01 \
  --frets_scale 0.02 \
  --seed 2024 \
  --epochs 100 \
  --model_id "T3Time_FreTS_FusionExp_best_reproduce"

echo ""
echo "=========================================="
echo "复现实验完成！"
echo "=========================================="
