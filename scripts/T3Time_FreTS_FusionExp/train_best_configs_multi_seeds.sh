#!/bin/bash
# T3Time_FreTS_Gated_Qwen 最佳配置批量训练脚本
# 使用两个最佳参数组合（最小MSE和最小MAE）在多个预测长度和种子上进行训练

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

# 创建日志目录
LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_BestConfigs/ETTh1"
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

# 两个最佳参数配置
# 配置1: 最小MSE组合
declare -A CONFIG1=(
    [channel]=64
    [dropout]=0.1
    [head]=16
    [name]="BestMSE"
)

# 配置2: 最小MAE组合
declare -A CONFIG2=(
    [channel]=64
    [dropout]=0.5
    [head]=4
    [name]="BestMAE"
)

# 预测长度列表
PRED_LENS=(96 192 336 720)

# 种子范围 (2020~2090)
SEEDS=()
for seed in $(seq 2020 2090); do
    SEEDS+=($seed)
done

# 计算总实验数
total_configs=2
total_pred_lens=${#PRED_LENS[@]}
total_seeds=${#SEEDS[@]}
total_exps=$((total_configs * total_pred_lens * total_seeds))
current=0

echo "=========================================="
echo "T3Time_FreTS_Gated_Qwen 最佳配置批量训练"
echo "=========================================="
echo "配置数量: ${total_configs}"
echo "  - 配置1 (最小MSE): Channel=${CONFIG1[channel]}, Dropout=${CONFIG1[dropout]}, Head=${CONFIG1[head]}"
echo "  - 配置2 (最小MAE): Channel=${CONFIG2[channel]}, Dropout=${CONFIG2[dropout]}, Head=${CONFIG2[head]}"
echo "预测长度: ${PRED_LENS[@]}"
echo "种子数量: ${total_seeds} (2020~2090)"
echo "总实验数: ${total_exps}"
echo "=========================================="
echo ""

# 遍历两个配置
for config_num in 1 2; do
    if [ $config_num -eq 1 ]; then
        CONFIG_NAME="${CONFIG1[name]}"
        CHANNEL="${CONFIG1[channel]}"
        DROPOUT="${CONFIG1[dropout]}"
        HEAD="${CONFIG1[head]}"
    else
        CONFIG_NAME="${CONFIG2[name]}"
        CHANNEL="${CONFIG2[channel]}"
        DROPOUT="${CONFIG2[dropout]}"
        HEAD="${CONFIG2[head]}"
    fi
    
    echo "=========================================="
    echo "开始配置: ${CONFIG_NAME} (Channel=${CHANNEL}, Dropout=${DROPOUT}, Head=${HEAD})"
    echo "=========================================="
    
    # 遍历预测长度
    for pred_len in "${PRED_LENS[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "预测长度: ${pred_len}"
        echo "----------------------------------------"
        
        # 遍历种子
        for seed in "${SEEDS[@]}"; do
            current=$((current + 1))
            
            # 生成模型ID
            MODEL_ID="T3Time_FreTS_Gated_Qwen_${CONFIG_NAME}_pred${pred_len}_seed${seed}"
            
            # 生成日志文件路径
            LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_pred${pred_len}_seed${seed}.log"
            
            echo "[${current}/${total_exps}] 训练: ${MODEL_ID}"
            echo "  日志: ${LOG_FILE}"
            
            # 运行训练
            python /root/0/T3Time/train_frets_gated_qwen.py \
                --data_path "${DATA_PATH}" \
                --batch_size "${BATCH_SIZE}" \
                --seq_len "${SEQ_LEN}" \
                --pred_len "${pred_len}" \
                --epochs "${EPOCHS}" \
                --es_patience "${PATIENCE}" \
                --seed "${seed}" \
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
                > "${LOG_FILE}" 2>&1
            
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                echo "  ✅ 完成 (退出码: ${exit_code})"
            else
                echo "  ⚠️  失败 (退出码: ${exit_code})，继续下一个实验..."
            fi
            
            # 短暂休息，避免GPU过载
            sleep 2
        done
    done
done

echo ""
echo "=========================================="
echo "✅ 所有实验完成！"
echo "=========================================="
echo "总实验数: ${total_exps}"
echo "日志目录: ${LOG_DIR}"
echo ""
echo "查看结果:"
echo "  python scripts/T3Time_FreTS_FusionExp/find_best_params_seed2088.py"
echo "  或直接查看: experiment_results.log"
echo "=========================================="
