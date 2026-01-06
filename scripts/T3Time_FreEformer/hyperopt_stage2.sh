#!/bin/bash
# T3Time_FreEformer_Gated_Qwen 阶段2参数寻优脚本
# 阶段2：训练参数寻优（learning_rate → dropout_n → batch_size）

set -uo pipefail

# 激活 conda 环境
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

# 设置环境变量
export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="/root/0/T3Time/Results/T3Time_FreEformer/Stage2"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

# 固定参数（基于阶段1的最佳结果）
DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=96
NUM_NODES=7
CHANNEL=32
FRE_E_LAYER=1
EMBED_SIZE=8
E_LAYER=1
D_LAYER=1
HEAD=8
EPOCHS=50
ES_PATIENCE=10
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
SEED=2021
WEIGHT_DECAY=1e-4
LOSS_FN="smooth_l1"
MODEL_ID_PREFIX="T3Time_FreEformer_Stage2"

echo "=========================================="
echo "T3Time_FreEformer_Gated_Qwen 阶段2参数寻优"
echo "=========================================="
echo "固定参数（阶段1最佳结果）:"
echo "  Channel: ${CHANNEL}"
echo "  Fre_E_Layer: ${FRE_E_LAYER}"
echo "  Embed_Size: ${EMBED_SIZE}"
echo "  E_Layer: ${E_LAYER}, D_Layer: ${D_LAYER}"
echo "  Head: ${HEAD}"
echo "  Seed: ${SEED}"
echo "=========================================="

# 步骤2.1: learning_rate 寻优
echo ""
echo "=========================================="
echo "步骤 2.1: Learning_Rate 寻优"
echo "=========================================="
LEARNING_RATES=(5e-5 7.5e-5 1e-4 1.5e-4)
DROPOUT_N=0.1
BATCH_SIZE=32

best_learning_rate=""
best_learning_rate_mse=999999.0

for learning_rate in "${LEARNING_RATES[@]}"; do
    MODEL_ID="${MODEL_ID_PREFIX}_Step2_1_LR${learning_rate}"
    LOG_FILE="${LOG_DIR}/${MODEL_ID}_${SEED}.log"
    
    echo ""
    echo "----------------------------------------"
    echo "实验: Learning_Rate=${learning_rate}, Dropout=${DROPOUT_N}, Batch_Size=${BATCH_SIZE}"
    echo "Model_ID: ${MODEL_ID}"
    echo "----------------------------------------"
    
    python -u train_freeformer_gated_qwen.py \
        --data_path "${DATA_PATH}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${PRED_LEN}" \
        --num_nodes "${NUM_NODES}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${learning_rate}" \
        --dropout_n "${DROPOUT_N}" \
        --channel "${CHANNEL}" \
        --e_layer "${E_LAYER}" \
        --d_layer "${D_LAYER}" \
        --head "${HEAD}" \
        --epochs "${EPOCHS}" \
        --es_patience "${ES_PATIENCE}" \
        --lradj "${LRADJ}" \
        --embed_version "${EMBED_VERSION}" \
        --seed "${SEED}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --loss_fn "${LOSS_FN}" \
        --model_id "${MODEL_ID}" \
        --embed_size "${EMBED_SIZE}" \
        --fre_e_layer "${FRE_E_LAYER}" \
        > "${LOG_FILE}" 2>&1
    
    if [ $? -eq 0 ]; then
        mse=$(grep "Test MSE:" "${LOG_FILE}" | tail -1 | awk '{print $NF}')
        if [ -n "${mse}" ] && [ "${mse}" != "0" ]; then
            echo "✅ Learning_Rate=${learning_rate}: MSE=${mse}"
            comparison=$(python3 -c "print(1 if ${mse} < ${best_learning_rate_mse} else 0)")
            if [ "${comparison}" = "1" ]; then
                best_learning_rate_mse=${mse}
                best_learning_rate=${learning_rate}
                echo "  🏆 新的最佳 Learning_Rate: ${best_learning_rate} (MSE: ${best_learning_rate_mse})"
            fi
        else
            echo "⚠️  未能从日志中提取 MSE"
        fi
    else
        echo "❌ 实验失败，查看日志: ${LOG_FILE}"
    fi
done

if [ -z "${best_learning_rate}" ]; then
    echo "❌ 未找到有效的 Learning_Rate 结果，退出"
    exit 1
fi

echo ""
echo "=========================================="
echo "步骤 2.1 完成: 最佳 Learning_Rate = ${best_learning_rate} (MSE: ${best_learning_rate_mse})"
echo "=========================================="

# 步骤2.2: dropout_n 寻优（固定最佳 learning_rate）
echo ""
echo "=========================================="
echo "步骤 2.2: Dropout 寻优（Learning_Rate=${best_learning_rate}）"
echo "=========================================="
DROPOUTS=(0.1 0.2 0.3 0.4 0.5)
BATCH_SIZE=32

best_dropout=""
best_dropout_mse=999999.0

for dropout_n in "${DROPOUTS[@]}"; do
    MODEL_ID="${MODEL_ID_PREFIX}_Step2_2_LR${best_learning_rate}_Dropout${dropout_n}"
    LOG_FILE="${LOG_DIR}/${MODEL_ID}_${SEED}.log"
    
    echo ""
    echo "----------------------------------------"
    echo "实验: Learning_Rate=${best_learning_rate}, Dropout=${dropout_n}, Batch_Size=${BATCH_SIZE}"
    echo "Model_ID: ${MODEL_ID}"
    echo "----------------------------------------"
    
    python -u train_freeformer_gated_qwen.py \
        --data_path "${DATA_PATH}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${PRED_LEN}" \
        --num_nodes "${NUM_NODES}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${best_learning_rate}" \
        --dropout_n "${dropout_n}" \
        --channel "${CHANNEL}" \
        --e_layer "${E_LAYER}" \
        --d_layer "${D_LAYER}" \
        --head "${HEAD}" \
        --epochs "${EPOCHS}" \
        --es_patience "${ES_PATIENCE}" \
        --lradj "${LRADJ}" \
        --embed_version "${EMBED_VERSION}" \
        --seed "${SEED}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --loss_fn "${LOSS_FN}" \
        --model_id "${MODEL_ID}" \
        --embed_size "${EMBED_SIZE}" \
        --fre_e_layer "${FRE_E_LAYER}" \
        > "${LOG_FILE}" 2>&1
    
    if [ $? -eq 0 ]; then
        mse=$(grep "Test MSE:" "${LOG_FILE}" | tail -1 | awk '{print $NF}')
        if [ -n "${mse}" ] && [ "${mse}" != "0" ]; then
            echo "✅ Dropout=${dropout_n}: MSE=${mse}"
            comparison=$(python3 -c "print(1 if ${mse} < ${best_dropout_mse} else 0)")
            if [ "${comparison}" = "1" ]; then
                best_dropout_mse=${mse}
                best_dropout=${dropout_n}
                echo "  🏆 新的最佳 Dropout: ${best_dropout} (MSE: ${best_dropout_mse})"
            fi
        else
            echo "⚠️  未能从日志中提取 MSE"
        fi
    else
        echo "❌ 实验失败，查看日志: ${LOG_FILE}"
    fi
done

if [ -z "${best_dropout}" ]; then
    echo "❌ 未找到有效的 Dropout 结果，退出"
    exit 1
fi

echo ""
echo "=========================================="
echo "步骤 2.2 完成: 最佳 Dropout = ${best_dropout} (MSE: ${best_dropout_mse})"
echo "=========================================="

# 步骤2.3: batch_size 寻优（固定前两者）
echo ""
echo "=========================================="
echo "步骤 2.3: Batch_Size 寻优（Learning_Rate=${best_learning_rate}, Dropout=${best_dropout}）"
echo "=========================================="
BATCH_SIZES=(16 32 64)

best_batch_size=""
best_batch_size_mse=999999.0

for batch_size in "${BATCH_SIZES[@]}"; do
    MODEL_ID="${MODEL_ID_PREFIX}_Step2_3_LR${best_learning_rate}_Dropout${best_dropout}_Batch${batch_size}"
    LOG_FILE="${LOG_DIR}/${MODEL_ID}_${SEED}.log"
    
    echo ""
    echo "----------------------------------------"
    echo "实验: Learning_Rate=${best_learning_rate}, Dropout=${best_dropout}, Batch_Size=${batch_size}"
    echo "Model_ID: ${MODEL_ID}"
    echo "----------------------------------------"
    
    python -u train_freeformer_gated_qwen.py \
        --data_path "${DATA_PATH}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${PRED_LEN}" \
        --num_nodes "${NUM_NODES}" \
        --batch_size "${batch_size}" \
        --learning_rate "${best_learning_rate}" \
        --dropout_n "${best_dropout}" \
        --channel "${CHANNEL}" \
        --e_layer "${E_LAYER}" \
        --d_layer "${D_LAYER}" \
        --head "${HEAD}" \
        --epochs "${EPOCHS}" \
        --es_patience "${ES_PATIENCE}" \
        --lradj "${LRADJ}" \
        --embed_version "${EMBED_VERSION}" \
        --seed "${SEED}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --loss_fn "${LOSS_FN}" \
        --model_id "${MODEL_ID}" \
        --embed_size "${EMBED_SIZE}" \
        --fre_e_layer "${FRE_E_LAYER}" \
        > "${LOG_FILE}" 2>&1
    
    if [ $? -eq 0 ]; then
        mse=$(grep "Test MSE:" "${LOG_FILE}" | tail -1 | awk '{print $NF}')
        if [ -n "${mse}" ] && [ "${mse}" != "0" ]; then
            echo "✅ Batch_Size=${batch_size}: MSE=${mse}"
            comparison=$(python3 -c "print(1 if ${mse} < ${best_batch_size_mse} else 0)")
            if [ "${comparison}" = "1" ]; then
                best_batch_size_mse=${mse}
                best_batch_size=${batch_size}
                echo "  🏆 新的最佳 Batch_Size: ${best_batch_size} (MSE: ${best_batch_size_mse})"
            fi
        else
            echo "⚠️  未能从日志中提取 MSE"
        fi
    else
        echo "❌ 实验失败，查看日志: ${LOG_FILE}"
    fi
done

if [ -z "${best_batch_size}" ]; then
    echo "❌ 未找到有效的 Batch_Size 结果，退出"
    exit 1
fi

echo ""
echo "=========================================="
echo "阶段2完成！"
echo "=========================================="
echo "最佳参数组合:"
echo "  Channel: ${CHANNEL}"
echo "  Fre_E_Layer: ${FRE_E_LAYER}"
echo "  Embed_Size: ${EMBED_SIZE}"
echo "  Learning_Rate: ${best_learning_rate}"
echo "  Dropout: ${best_dropout}"
echo "  Batch_Size: ${best_batch_size}"
echo "  最终 MSE: ${best_batch_size_mse}"
echo "=========================================="
echo ""
echo "所有结果已保存到: ${RESULT_LOG}"
echo "日志文件保存在: ${LOG_DIR}"
echo ""
echo "运行分析脚本查看详细结果:"
echo "  python scripts/T3Time_FreEformer/analyze_stage2_results.py"
