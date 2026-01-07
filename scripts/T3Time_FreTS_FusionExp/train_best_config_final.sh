#!/bin/bash
# T3Time_FreTS_Gated_Qwen 最佳配置训练脚本
# 使用简化版本模型（不带消融选项）

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
DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=96
CHANNEL=64
BATCH_SIZE=16
LEARNING_RATE=0.0001
DROPOUT_N=0.1
WEIGHT_DECAY=1e-4
E_LAYER=1
D_LAYER=1
HEAD=8
EPOCHS=100
PATIENCE=10
LOSS_FN="smooth_l1"
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
MODEL_ID="T3Time_FreTS_Gated_Qwen"

# 运行多个种子（2020-2090）
for SEED in {2020..2090}; do
    echo "=========================================="
    echo "训练开始 - Seed: ${SEED}"
    echo "=========================================="
    
    python /root/0/T3Time/train_frets_gated_qwen.py \
        --data_path "${DATA_PATH}" \
        --batch_size "${BATCH_SIZE}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${PRED_LEN}" \
        --epochs "${EPOCHS}" \
        --es_patience "${PATIENCE}" \
        --seed "${SEED}" \
        --channel "${CHANNEL}" \
        --learning_rate "${LEARNING_RATE}" \
        --dropout_n "${DROPOUT_N}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --e_layer "${E_LAYER}" \
        --d_layer "${D_LAYER}" \
        --head "${HEAD}" \
        --loss_fn "${LOSS_FN}" \
        --lradj "${LRADJ}" \
        --embed_version "${EMBED_VERSION}" \
        --model_id "${MODEL_ID}" \
        || true  # 允许单个实验失败时继续运行
    
    echo "✅ Seed ${SEED} 完成"
    echo ""
done

echo "=========================================="
echo "✅ 所有训练完成！"
echo "=========================================="
echo "结果已追加到: /root/0/T3Time/experiment_results.log"
