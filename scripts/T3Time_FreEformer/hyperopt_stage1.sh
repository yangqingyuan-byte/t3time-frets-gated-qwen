#!/bin/bash
# T3Time_FreEformer_Gated_Qwen 阶段1参数寻优脚本
# 阶段1：架构参数寻优（channel → fre_e_layer → embed_size）

set -uo pipefail

# 激活 conda 环境
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

# 设置环境变量
export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="/root/0/T3Time/Results/T3Time_FreEformer/Stage1"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

# 固定参数
DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=96
NUM_NODES=7
BATCH_SIZE=32
LEARNING_RATE=1e-4
DROPOUT_N=0.1
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
MODEL_ID_PREFIX="T3Time_FreEformer_Stage1"

echo "=========================================="
echo "T3Time_FreEformer_Gated_Qwen 阶段1参数寻优"
echo "=========================================="
echo "固定参数:"
echo "  Data: ${DATA_PATH}"
echo "  Seq_Len: ${SEQ_LEN}, Pred_Len: ${PRED_LEN}"
echo "  Learning_Rate: ${LEARNING_RATE}"
echo "  Dropout: ${DROPOUT_N}"
echo "  Batch_Size: ${BATCH_SIZE}"
echo "  E_Layer: ${E_LAYER}, D_Layer: ${D_LAYER}"
echo "  Head: ${HEAD}"
echo "  Seed: ${SEED}"
echo "=========================================="

# 步骤1.1: channel 寻优
echo ""
echo "=========================================="
echo "步骤 1.1: Channel 寻优"
echo "=========================================="
CHANNELS=(32 64 96 128)
FRE_E_LAYER=1
EMBED_SIZE=16

best_channel=""
best_channel_mse=999999.0

for channel in "${CHANNELS[@]}"; do
    # 检查 head 是否整除 channel
    if [ $((channel % HEAD)) -ne 0 ]; then
        echo "⚠️  跳过 channel=${channel}（不能被 head=${HEAD} 整除）"
        continue
    fi
    
    MODEL_ID="${MODEL_ID_PREFIX}_Step1_1_Channel${channel}"
    LOG_FILE="${LOG_DIR}/${MODEL_ID}_${SEED}.log"
    
    echo ""
    echo "----------------------------------------"
    echo "实验: Channel=${channel}, Fre_E_Layer=${FRE_E_LAYER}, Embed_Size=${EMBED_SIZE}"
    echo "Model_ID: ${MODEL_ID}"
    echo "----------------------------------------"
    
    python -u train_freeformer_gated_qwen.py \
        --data_path "${DATA_PATH}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${PRED_LEN}" \
        --num_nodes "${NUM_NODES}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LEARNING_RATE}" \
        --dropout_n "${DROPOUT_N}" \
        --channel "${channel}" \
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
        # 从日志中提取 MSE
        mse=$(grep "Test MSE:" "${LOG_FILE}" | tail -1 | awk '{print $NF}')
        if [ -n "${mse}" ] && [ "${mse}" != "0" ]; then
            echo "✅ Channel=${channel}: MSE=${mse}"
            # 比较 MSE（越小越好）- 使用 Python 进行浮点数比较
            comparison=$(python3 -c "print(1 if ${mse} < ${best_channel_mse} else 0)")
            if [ "${comparison}" = "1" ]; then
                best_channel_mse=${mse}
                best_channel=${channel}
                echo "  🏆 新的最佳 Channel: ${best_channel} (MSE: ${best_channel_mse})"
            fi
        else
            echo "⚠️  未能从日志中提取 MSE"
        fi
    else
        echo "❌ 实验失败，查看日志: ${LOG_FILE}"
    fi
done

if [ -z "${best_channel}" ]; then
    echo "❌ 未找到有效的 Channel 结果，退出"
    exit 1
fi

echo ""
echo "=========================================="
echo "步骤 1.1 完成: 最佳 Channel = ${best_channel} (MSE: ${best_channel_mse})"
echo "=========================================="

# 步骤1.2: fre_e_layer 寻优（固定最佳 channel）
echo ""
echo "=========================================="
echo "步骤 1.2: Fre_E_Layer 寻优（Channel=${best_channel}）"
echo "=========================================="
FRE_E_LAYERS=(1 2 3)
EMBED_SIZE=16

best_fre_e_layer=""
best_fre_e_layer_mse=999999.0

for fre_e_layer in "${FRE_E_LAYERS[@]}"; do
    MODEL_ID="${MODEL_ID_PREFIX}_Step1_2_Channel${best_channel}_FreELayer${fre_e_layer}"
    LOG_FILE="${LOG_DIR}/${MODEL_ID}_${SEED}.log"
    
    echo ""
    echo "----------------------------------------"
    echo "实验: Channel=${best_channel}, Fre_E_Layer=${fre_e_layer}, Embed_Size=${EMBED_SIZE}"
    echo "Model_ID: ${MODEL_ID}"
    echo "----------------------------------------"
    
    python -u train_freeformer_gated_qwen.py \
        --data_path "${DATA_PATH}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${PRED_LEN}" \
        --num_nodes "${NUM_NODES}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LEARNING_RATE}" \
        --dropout_n "${DROPOUT_N}" \
        --channel "${best_channel}" \
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
        --fre_e_layer "${fre_e_layer}" \
        > "${LOG_FILE}" 2>&1
    
    if [ $? -eq 0 ]; then
        mse=$(grep "Test MSE:" "${LOG_FILE}" | tail -1 | awk '{print $NF}')
        if [ -n "${mse}" ]; then
            echo "✅ Fre_E_Layer=${fre_e_layer}: MSE=${mse}"
            comparison=$(python3 -c "print(1 if ${mse} < ${best_fre_e_layer_mse} else 0)")
            if [ "${comparison}" = "1" ]; then
                best_fre_e_layer_mse=${mse}
                best_fre_e_layer=${fre_e_layer}
                echo "  🏆 新的最佳 Fre_E_Layer: ${best_fre_e_layer} (MSE: ${best_fre_e_layer_mse})"
            fi
        else
            echo "⚠️  未能从日志中提取 MSE"
        fi
    else
        echo "❌ 实验失败，查看日志: ${LOG_FILE}"
    fi
done

if [ -z "${best_fre_e_layer}" ]; then
    echo "❌ 未找到有效的 Fre_E_Layer 结果，退出"
    exit 1
fi

echo ""
echo "=========================================="
echo "步骤 1.2 完成: 最佳 Fre_E_Layer = ${best_fre_e_layer} (MSE: ${best_fre_e_layer_mse})"
echo "=========================================="

# 步骤1.3: embed_size 寻优（固定前两者）
echo ""
echo "=========================================="
echo "步骤 1.3: Embed_Size 寻优（Channel=${best_channel}, Fre_E_Layer=${best_fre_e_layer}）"
echo "=========================================="
EMBED_SIZES=(8 16 32)

best_embed_size=""
best_embed_size_mse=999999.0

for embed_size in "${EMBED_SIZES[@]}"; do
    MODEL_ID="${MODEL_ID_PREFIX}_Step1_3_Channel${best_channel}_FreELayer${best_fre_e_layer}_EmbedSize${embed_size}"
    LOG_FILE="${LOG_DIR}/${MODEL_ID}_${SEED}.log"
    
    echo ""
    echo "----------------------------------------"
    echo "实验: Channel=${best_channel}, Fre_E_Layer=${best_fre_e_layer}, Embed_Size=${embed_size}"
    echo "Model_ID: ${MODEL_ID}"
    echo "----------------------------------------"
    
    python -u train_freeformer_gated_qwen.py \
        --data_path "${DATA_PATH}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${PRED_LEN}" \
        --num_nodes "${NUM_NODES}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LEARNING_RATE}" \
        --dropout_n "${DROPOUT_N}" \
        --channel "${best_channel}" \
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
        --embed_size "${embed_size}" \
        --fre_e_layer "${best_fre_e_layer}" \
        > "${LOG_FILE}" 2>&1
    
    if [ $? -eq 0 ]; then
        mse=$(grep "Test MSE:" "${LOG_FILE}" | tail -1 | awk '{print $NF}')
        if [ -n "${mse}" ]; then
            echo "✅ Embed_Size=${embed_size}: MSE=${mse}"
            comparison=$(python3 -c "print(1 if ${mse} < ${best_embed_size_mse} else 0)")
            if [ "${comparison}" = "1" ]; then
                best_embed_size_mse=${mse}
                best_embed_size=${embed_size}
                echo "  🏆 新的最佳 Embed_Size: ${best_embed_size} (MSE: ${best_embed_size_mse})"
            fi
        else
            echo "⚠️  未能从日志中提取 MSE"
        fi
    else
        echo "❌ 实验失败，查看日志: ${LOG_FILE}"
    fi
done

if [ -z "${best_embed_size}" ]; then
    echo "❌ 未找到有效的 Embed_Size 结果，退出"
    exit 1
fi

echo ""
echo "=========================================="
echo "阶段1完成！"
echo "=========================================="
echo "最佳参数组合:"
echo "  Channel: ${best_channel}"
echo "  Fre_E_Layer: ${best_fre_e_layer}"
echo "  Embed_Size: ${best_embed_size}"
echo "  最终 MSE: ${best_embed_size_mse}"
echo "=========================================="
echo ""
echo "所有结果已保存到: ${RESULT_LOG}"
echo "日志文件保存在: ${LOG_DIR}"
echo ""
echo "运行分析脚本查看详细结果:"
echo "  python scripts/T3Time_FreEformer/analyze_stage1_results.py"
