#!/bin/bash
# T3Time_FreTS_Gated_Qwen_Hyperopt 扩展参数寻优脚本
# 针对 pred_len=720, seed=2088，重点优化 MSE
# 扩展参数：learning_rate, weight_decay, loss_fn, batch_size

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
E_LAYER=1
D_LAYER=1
EPOCHS=100
PATIENCE=10
SEED=2088
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
MODEL_ID="T3Time_FreTS_Gated_Qwen_Hyperopt"

# 基于之前最佳结果的参数（作为起点）
BASE_CHANNEL=64
BASE_DROPOUT=0.50
BASE_HEAD=8

# 扩展参数搜索空间（重点优化 MSE）
# 1. 学习率：较小的学习率可能有助于减少大误差
LEARNING_RATES=(5e-5 7.5e-5 1e-4 1.5e-4)

# 2. 权重衰减：更强的正则化可能有助于减少大误差（MSE 对大误差敏感）
WEIGHT_DECAYS=(1e-4 5e-4 1e-3 2e-3)

# 3. 损失函数：MSE loss 直接优化 MSE 指标
LOSS_FNS=("mse" "smooth_l1")

# 4. 批次大小：可能影响训练稳定性
BATCH_SIZES=(16 32)

# 计算总实验数
total_exps=$((${#LEARNING_RATES[@]} * ${#WEIGHT_DECAYS[@]} * ${#LOSS_FNS[@]} * ${#BATCH_SIZES[@]}))
current=0

echo "=========================================="
echo "T3Time_FreTS_Gated_Qwen_Hyperopt 扩展参数寻优"
echo "Seed: ${SEED}, Pred_Len: ${PRED_LEN}"
echo "基础参数: Channel=${BASE_CHANNEL}, Dropout=${BASE_DROPOUT}, Head=${BASE_HEAD}"
echo "扩展搜索空间:"
echo "  Learning Rate: ${LEARNING_RATES[@]}"
echo "  Weight Decay: ${WEIGHT_DECAYS[@]}"
echo "  Loss Function: ${LOSS_FNS[@]}"
echo "  Batch Size: ${BATCH_SIZES[@]}"
echo "总实验数: ${total_exps}"
echo "=========================================="
echo ""

# 网格搜索
for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
        for LOSS_FN in "${LOSS_FNS[@]}"; do
            for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                current=$((current + 1))
                
                echo "[${current}/${total_exps}] 开始实验..."
                echo "  LR: ${LEARNING_RATE}, WD: ${WEIGHT_DECAY}, Loss: ${LOSS_FN}, BS: ${BATCH_SIZE}"
                
                log_file="${LOG_DIR}/pred${PRED_LEN}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_loss${LOSS_FN}_bs${BATCH_SIZE}_seed${SEED}.log"
                
                # 运行训练
                python /root/0/T3Time/train_frets_gated_qwen.py \
                    --data_path "${DATA_PATH}" \
                    --batch_size "${BATCH_SIZE}" \
                    --seq_len "${SEQ_LEN}" \
                    --pred_len "${PRED_LEN}" \
                    --epochs "${EPOCHS}" \
                    --es_patience "${PATIENCE}" \
                    --seed "${SEED}" \
                    --channel "${BASE_CHANNEL}" \
                    --learning_rate "${LEARNING_RATE}" \
                    --dropout_n "${BASE_DROPOUT}" \
                    --weight_decay "${WEIGHT_DECAY}" \
                    --e_layer "${E_LAYER}" \
                    --d_layer "${D_LAYER}" \
                    --head "${BASE_HEAD}" \
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
done

echo "=========================================="
echo "✅ 所有扩展参数寻优实验完成！"
echo "=========================================="
echo "结果已追加到: ${RESULT_LOG}"
echo "日志文件保存在: ${LOG_DIR}"
echo ""
echo "运行分析脚本查看最佳参数组合:"
echo "  python scripts/T3Time_FreTS_FusionExp/find_best_hyperopt_pred720_seed2088.py"
echo "=========================================="
