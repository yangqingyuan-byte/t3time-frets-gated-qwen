#!/bin/bash
# T3Time_FreTS_Gated_Qwen 单次训练示例
# 使用最佳配置参数（seed=2088 的配置）

set -uo pipefail

# 清理可能的环境变量问题
unset __vsc_prompt_cmd_original 2>/dev/null || true

# 激活 conda 环境
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

# 设置环境变量
export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

# 最佳配置参数（基于 seed=2088 的结果）
python /root/0/T3Time/train_frets_gated_qwen.py \
    --data_path ETTh1 \
    --batch_size 16 \
    --seq_len 96 \
    --pred_len 96 \
    --epochs 100 \
    --es_patience 10 \
    --seed 2088 \
    --channel 64 \
    --learning_rate 0.0001 \
    --dropout_n 0.1 \
    --weight_decay 1e-4 \
    --e_layer 1 \
    --d_layer 1 \
    --head 8 \
    --loss_fn smooth_l1 \
    --lradj type1 \
    --embed_version qwen3_0.6b \
    --model_id T3Time_FreTS_Gated_Qwen

echo "✅ 训练完成！"
