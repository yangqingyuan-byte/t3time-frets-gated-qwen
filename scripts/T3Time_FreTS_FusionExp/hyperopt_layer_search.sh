#!/bin/bash
# T3Time_FreTS_Gated_Qwen 编码器和解码器层数寻优脚本
# 基于最佳MSE参数组合，只搜索 e_layer 和 d_layer

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

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_LayerSearch/ETTh1"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

# 固定参数（基于最佳MSE参数组合）
DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=96
CHANNEL=64
HEAD=16
LEARNING_RATE=0.0001
WEIGHT_DECAY=0.0001
DROPOUT_N=0.1
BATCH_SIZE=16
LOSS_FN="smooth_l1"
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
EPOCHS=100
PATIENCE=10
SEED=2088
MODEL_ID="T3Time_FreTS_Gated_Qwen_LayerSearch"

# 搜索空间：编码器和解码器层数
E_LAYERS=(1 2 3)
D_LAYERS=(1 2 3)

# 计算总实验数
total_exps=$((${#E_LAYERS[@]} * ${#D_LAYERS[@]}))
current=0

echo "=========================================="
echo "T3Time_FreTS_Gated_Qwen 编码器和解码器层数寻优"
echo "=========================================="
echo "固定参数（基于最佳MSE组合）:"
echo "  Channel:        ${CHANNEL}"
echo "  Head:           ${HEAD}"
echo "  Learning_Rate:  ${LEARNING_RATE}"
echo "  Weight_Decay:   ${WEIGHT_DECAY}"
echo "  Dropout:        ${DROPOUT_N}"
echo "  Batch_Size:     ${BATCH_SIZE}"
echo "  Loss_Function:  ${LOSS_FN}"
echo "  Seed:           ${SEED}"
echo ""
echo "搜索空间:"
echo "  E_Layer (编码器层数): ${E_LAYERS[@]}"
echo "  D_Layer (解码器层数): ${D_LAYERS[@]}"
echo "总实验数: ${total_exps}"
echo "=========================================="
echo ""

# 网格搜索
for E_LAYER in "${E_LAYERS[@]}"; do
    for D_LAYER in "${D_LAYERS[@]}"; do
        current=$((current + 1))
        
        echo "[${current}/${total_exps}] 开始实验..."
        echo "  E_Layer: ${E_LAYER}, D_Layer: ${D_LAYER}"
        
        log_file="${LOG_DIR}/e${E_LAYER}_d${D_LAYER}_seed${SEED}.log"
        
        # 运行训练
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
            > "${log_file}" 2>&1 || true
        
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            mse=$(grep "Test MSE:" "${log_file}" | tail -1 | awk '{print $NF}')
            mae=$(grep "Test MAE:" "${log_file}" | tail -1 | awk '{print $NF}')
            if [ -n "${mse}" ] && [ "${mse}" != "0" ]; then
                echo "  ✅ 完成: MSE=${mse}, MAE=${mae}"
            else
                echo "  ✅ 完成（但未能提取MSE）"
            fi
        else
            echo "  ⚠️  失败 (退出码: ${exit_code})"
        fi
        echo ""
        
        sleep 2
    done
done

echo "=========================================="
echo "✅ 所有层数寻优实验完成！"
echo "=========================================="
echo "结果已追加到: ${RESULT_LOG}"
echo "日志文件保存在: ${LOG_DIR}"
echo ""
echo "运行分析脚本查看结果:"
echo "  python scripts/T3Time_FreTS_FusionExp/analyze_layer_search_results.py"
echo "=========================================="
