#!/bin/bash
# 使用最佳参数组合在多种子（2020-2090）上训练
# 最佳参数：Channel=64, Dropout=0.5, Head=8, Batch_Size=32, LR=7.5e-5, WD=0.0005

set -uo pipefail

# 清理可能的环境变量问题
unset __vsc_prompt_cmd_original 2>/dev/null || true

# 激活 conda 环境
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

# 设置环境变量
export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_BestParams_MultiSeed/ETTh1"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

# 固定参数（基于最佳参数组合）
DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=720
E_LAYER=1
D_LAYER=1
EPOCHS=150
PATIENCE=10
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
MODEL_ID="T3Time_FreTS_Gated_Qwen_BestParams_MultiSeed"

# 最佳参数组合
CHANNEL=64
DROPOUT_N=0.5
HEAD=8
BATCH_SIZE=32
LEARNING_RATE=7.5e-5
WEIGHT_DECAY=0.0005
LOSS_FN="smooth_l1"

# 种子范围：2020-2090（共71个种子）
SEED_START=2020
SEED_END=2090
SEEDS=($(seq ${SEED_START} ${SEED_END}))

total_exps=${#SEEDS[@]}
current=0

echo "=========================================="
echo "使用最佳参数组合进行多种子训练"
echo "预测长度: ${PRED_LEN}"
echo "种子范围: ${SEED_START}-${SEED_END} (共 ${total_exps} 个种子)"
echo ""
echo "最佳参数组合:"
echo "  Channel:     ${CHANNEL}"
echo "  Dropout:     ${DROPOUT_N}"
echo "  Head:        ${HEAD}"
echo "  Batch Size:  ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Weight Decay:  ${WEIGHT_DECAY}"
echo "  Loss Function: ${LOSS_FN}"
echo "=========================================="
echo ""

for SEED in "${SEEDS[@]}"; do
    current=$((current + 1))
    
    log_file="${LOG_DIR}/pred${PRED_LEN}_seed${SEED}.log"
    
    echo "[${current}/${total_exps}] 训练: seed=${SEED}"
    echo "  日志: ${log_file}"
    
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
        > "${log_file}" 2>&1
    
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
echo "✅ 所有多种子训练完成！"
echo "=========================================="
echo "结果已追加到: ${RESULT_LOG}"
echo "日志文件保存在: ${LOG_DIR}"
echo ""
echo "运行分析脚本查看结果:"
echo "  python scripts/T3Time_FreTS_FusionExp/analyze_best_params_multi_seed.py"
echo "=========================================="
