#!/bin/bash
# T3Time_FreTS_Gated_Qwen_LongSeq_v2 长序列预测训练脚本
# 针对 192, 336, 720 三个预测长度

set -uo pipefail

unset __vsc_prompt_cmd_original 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_LongSeq_v2/ETTh1"
mkdir -p "${LOG_DIR}"

# 固定参数
DATA_PATH="ETTh1"
SEQ_LEN=96
BATCH_SIZE=16
LEARNING_RATE=0.0001
WEIGHT_DECAY=1e-4
E_LAYER=1
D_LAYER=1
EPOCHS=100
PATIENCE=10
LOSS_FN="smooth_l1"
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"

# 最佳配置参数
CHANNEL=64
DROPOUT=0.1
HEAD=8
SEED=2088

# 长序列预测长度
PRED_LENS=(192 336 720)

USE_DYNAMIC_SPARSITY=1

total_exps=${#PRED_LENS[@]}
current=0

echo "=========================================="
echo "T3Time_FreTS_Gated_Qwen_LongSeq_v2 训练"
echo "预测长度: ${PRED_LENS[@]}"
echo "总实验数: ${total_exps}"
echo "=========================================="
echo ""

for pred_len in "${PRED_LENS[@]}"; do
    current=$((current + 1))
    
    MODEL_ID="T3Time_FreTS_Gated_Qwen_LongSeq_v2_pred${pred_len}_seed${SEED}"
    LOG_FILE="${LOG_DIR}/v2_pred${pred_len}_seed${SEED}.log"
    
    echo "[${current}/${total_exps}] 训练: pred_len=${pred_len}"
    echo "  日志: ${LOG_FILE}"
    
    python /root/0/T3Time/train_frets_gated_qwen_longseq_v2.py \
        --data_path "${DATA_PATH}" \
        --batch_size "${BATCH_SIZE}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${pred_len}" \
        --epochs "${EPOCHS}" \
        --es_patience "${PATIENCE}" \
        --seed "${SEED}" \
        --channel "${CHANNEL}" \
        --learning_rate "${LEARNING_RATE}" \
        --dropout_n "${DROPOUT}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --e_layer "${E_LAYER}" \
        --d_layer "${D_LAYER}" \
        --head "${HEAD}" \
        --loss_fn "${LOSS_FN}" \
        --lradj "${LRADJ}" \
        --embed_version "${EMBED_VERSION}" \
        --model_id "${MODEL_ID}" \
        --sparsity_threshold 0.009 \
        --frets_scale 0.018 \
        --use_dynamic_sparsity "${USE_DYNAMIC_SPARSITY}" \
        > "${LOG_FILE}" 2>&1
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "  ✅ 完成"
    else
        echo "  ⚠️  失败 (退出码: ${exit_code})"
    fi
    
    sleep 2
done

echo ""
echo "=========================================="
echo "✅ 所有实验完成！"
echo "=========================================="
echo "查看结果: python scripts/T3Time_FreTS_FusionExp/analyze_longseq_v2_results.py"
echo "=========================================="
