#!/bin/bash
# 原版 T3Time 多预测长度训练脚本
# 针对 96, 192, 336, 720 四个预测长度

set -uo pipefail

unset __vsc_prompt_cmd_original 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

LOG_DIR="/root/0/T3Time/Results/T3Time_Original/ETTh1"
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
EMBED_VERSION="qwen3_0.6b"

# 配置参数
CHANNEL=64
DROPOUT=0.1
HEAD=8
SEED=2088
D_LLM=1024

# 预测长度列表
PRED_LENS=(96 192 336 720)

total_exps=${#PRED_LENS[@]}
current=0

echo "=========================================="
echo "原版 T3Time 多预测长度训练"
echo "预测长度: ${PRED_LENS[@]}"
echo "总实验数: ${total_exps}"
echo "=========================================="
echo ""

for pred_len in "${PRED_LENS[@]}"; do
    current=$((current + 1))
    
    MODEL_ID="T3Time_Original_pred${pred_len}_seed${SEED}"
    LOG_FILE="${LOG_DIR}/original_pred${pred_len}_seed${SEED}.log"
    
    echo "[${current}/${total_exps}] 训练: pred_len=${pred_len}"
    echo "  日志: ${LOG_FILE}"
    
    python /root/0/T3Time/train.py \
        --device cuda \
        --data_path "${DATA_PATH}" \
        --channel "${CHANNEL}" \
        --num_nodes 7 \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${pred_len}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LEARNING_RATE}" \
        --dropout_n "${DROPOUT}" \
        --d_llm "${D_LLM}" \
        --e_layer "${E_LAYER}" \
        --d_layer "${D_LAYER}" \
        --head "${HEAD}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --epochs "${EPOCHS}" \
        --es_patience "${PATIENCE}" \
        --seed "${SEED}" \
        --embed_version "${EMBED_VERSION}" \
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
echo "查看结果: python scripts/T3Time_FreTS_FusionExp/compare_t3time_models.py"
echo "=========================================="
