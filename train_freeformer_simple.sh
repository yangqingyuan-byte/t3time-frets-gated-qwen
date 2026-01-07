#!/bin/bash
# T3Time_FreEformer_Gated_Qwen 简化版训练脚本
# 使用默认参数，避免维度问题

python -u train_freeformer_gated_qwen.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --num_nodes 7 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --dropout_n 0.2 \
  --channel 64 \
  --e_layer 1 \
  --d_layer 1 \
  --head 8 \
  --epochs 30 \
  --es_patience 8 \
  --lradj type1 \
  --embed_version qwen3_0.6b \
  --seed 2021 \
  --weight_decay 1e-4 \
  --loss_fn smooth_l1 \
  --model_id T3Time_FreEformer_Gated_Qwen_ETTh1_96 \
  --embed_size 16 \
  --fre_e_layer 1
