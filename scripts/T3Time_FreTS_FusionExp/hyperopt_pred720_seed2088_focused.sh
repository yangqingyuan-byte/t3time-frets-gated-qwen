#!/bin/bash
# T3Time_FreTS_Gated_Qwen_Hyperopt 聚焦参数寻优脚本
# 针对 pred_len=720, seed=2088，重点优化 MSE
# 聚焦在原版 T3Time 和小波包模型成功的参数范围内

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

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_Hyperopt/ETTh1"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

# 固定参数
DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=720
E_LAYER=1
D_LAYER=1
EPOCHS=150
PATIENCE=10
SEED=2088
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
MODEL_ID="T3Time_FreTS_Gated_Qwen_Hyperopt"

# 聚焦搜索空间（基于原版 T3Time 和小波包模型的成功经验）
# 原版 T3Time 最佳: channel=128, dropout=0.6, lr=7e-05, batch_size=32
# 当前最佳: channel=64, dropout=0.5, lr=7.5e-05, wd=0.0005

# 1. Channel: 重点测试更大的 channel（原版用 128）
CHANNELS=(96 128)

# 2. Dropout: 重点测试更高的 dropout（原版用 0.6）
DROPOUTS=(0.55 0.6 0.65)

# 3. Head: 保持合理范围
HEADS=(8 10 12)

# 4. Learning Rate: 参考原版的 7e-05
LEARNING_RATES=(6e-5 7e-5 7.5e-5 8e-5)

# 5. Weight Decay: 测试更强的正则化
WEIGHT_DECAYS=(5e-4 1e-3 2e-3)

# 6. Loss Function: 使用 MSE（与原版对齐，直接优化 MSE）
LOSS_FNS=("mse")

# 7. Batch Size: 参考原版的 32
BATCH_SIZES=(32)

# 计算总实验数
total_exps=$((${#CHANNELS[@]} * ${#DROPOUTS[@]} * ${#HEADS[@]} * ${#LEARNING_RATES[@]} * ${#WEIGHT_DECAYS[@]} * ${#LOSS_FNS[@]} * ${#BATCH_SIZES[@]}))
current=0

echo "=========================================="
echo "T3Time_FreTS_Gated_Qwen_Hyperopt 聚焦参数寻优"
echo "Seed: ${SEED}, Pred_Len: ${PRED_LEN}"
echo "目标: 超越当前最佳 MSE 0.462425，接近原版 T3Time 的 0.438817"
echo ""
echo "聚焦搜索空间（基于原版 T3Time 成功经验）:"
echo "  Channel: ${CHANNELS[@]} (原版用 128)"
echo "  Dropout: ${DROPOUTS[@]} (原版用 0.6)"
echo "  Head: ${HEADS[@]}"
echo "  Learning Rate: ${LEARNING_RATES[@]} (原版用 7e-05)"
echo "  Weight Decay: ${WEIGHT_DECAYS[@]}"
echo "  Loss Function: ${LOSS_FNS[@]} (MSE，与原版对齐)"
echo "  Batch Size: ${BATCH_SIZES[@]} (原版用 32)"
echo "总实验数: ${total_exps}"
echo "=========================================="
echo ""

# 网格搜索
for CHANNEL in "${CHANNELS[@]}"; do
    for DROPOUT_N in "${DROPOUTS[@]}"; do
        for HEAD in "${HEADS[@]}"; do
            for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
                for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                    for LOSS_FN in "${LOSS_FNS[@]}"; do
                        for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                            current=$((current + 1))
                            
                            echo "[${current}/${total_exps}] 开始实验..."
                            echo "  Channel: ${CHANNEL}, Dropout: ${DROPOUT_N}, Head: ${HEAD}"
                            echo "  LR: ${LEARNING_RATE}, WD: ${WEIGHT_DECAY}, Loss: ${LOSS_FN}, BS: ${BATCH_SIZE}"
                            
                            log_file="${LOG_DIR}/pred${PRED_LEN}_c${CHANNEL}_d${DROPOUT_N}_h${HEAD}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_loss${LOSS_FN}_bs${BATCH_SIZE}_seed${SEED}.log"
                            
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
            done
        done
    done
done

echo "=========================================="
echo "✅ 所有聚焦参数寻优实验完成！"
echo "=========================================="
echo "结果已追加到: ${RESULT_LOG}"
echo "日志文件保存在: ${LOG_DIR}"
echo ""
echo "运行分析脚本查看最佳参数组合:"
echo "  python scripts/T3Time_FreTS_FusionExp/find_best_hyperopt_pred720_seed2088.py"
echo "=========================================="
