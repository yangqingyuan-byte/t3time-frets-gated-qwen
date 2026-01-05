#!/bin/bash
# T3Time_FreTS_Gated_Qwen_Hyperopt 参数寻优脚本
# 在种子 2088 和预测长度 720 上进行参数寻优

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

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_Hyperopt/ETTh1"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

# 固定参数
DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=720
BATCH_SIZE=16
LEARNING_RATE=0.0001
WEIGHT_DECAY=1e-4
E_LAYER=1
D_LAYER=1
EPOCHS=100
PATIENCE=10
SEED=2088
LOSS_FN="smooth_l1"
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
MODEL_ID="T3Time_FreTS_Gated_Qwen_Hyperopt"

# 参数搜索空间（针对 pred_len=720 优化）
CHANNELS=(52 56 60 64)
DROPOUTS=(0.40 0.45 0.50 0.55 0.60)
HEADS=(5 6 7 8 9 10 11 12)

# 计算总实验数
total_exps=$((${#CHANNELS[@]} * ${#DROPOUTS[@]} * ${#HEADS[@]}))
current=0

echo "=========================================="
echo "T3Time_FreTS_Gated_Qwen_Hyperopt 参数寻优"
echo "Seed: ${SEED}, Pred_Len: ${PRED_LEN}"
echo "搜索空间:"
echo "  Channel: ${CHANNELS[@]}"
echo "  Dropout: ${DROPOUTS[@]}"
echo "  Head: ${HEADS[@]}"
echo "总实验数: ${total_exps}"
echo "=========================================="
echo ""

# 网格搜索
for CHANNEL in "${CHANNELS[@]}"; do
    for DROPOUT_N in "${DROPOUTS[@]}"; do
        for HEAD in "${HEADS[@]}"; do
            current=$((current + 1))
            
            echo "[${current}/${total_exps}] 开始实验..."
            echo "  Channel: ${CHANNEL}, Dropout: ${DROPOUT_N}, Head: ${HEAD}"
            
            log_file="${LOG_DIR}/pred${PRED_LEN}_channel${CHANNEL}_dropout${DROPOUT_N}_head${HEAD}_seed${SEED}.log"
            
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
                echo "  ✅ 完成: ${log_file}"
            else
                echo "  ⚠️  失败 (退出码: ${exit_code}): ${log_file}"
            fi
            echo ""
            
            sleep 2
        done
    done
done

echo "=========================================="
echo "✅ 所有参数寻优实验完成！"
echo "=========================================="
echo "结果已追加到: ${RESULT_LOG}"
echo "日志文件保存在: ${LOG_DIR}"
echo ""
echo "运行分析脚本查看最佳参数组合:"
echo "  python scripts/T3Time_FreTS_FusionExp/find_best_hyperopt_pred720_seed2088.py"
echo "=========================================="
