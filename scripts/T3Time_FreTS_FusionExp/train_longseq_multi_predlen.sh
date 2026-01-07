#!/bin/bash
# T3Time_FreTS_Gated_Qwen_LongSeq 多预测长度训练脚本
# 在多个预测长度上测试改进模型的效果

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
LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_LongSeq/ETTh1"
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

# 测试配置
CHANNEL=64
DROPOUT=0.1
HEAD=8
SEED=2088

# 预测长度列表
PRED_LENS=(96 192 336 720)

# 改进配置（可以测试不同组合）
HORIZON_NORMS=("div1000")  # div1000, div100, log
FUSION_MODES=("weighted_avg")  # weighted_avg, residual
USE_DYNAMIC_SPARSITY=1  # 1=启用, 0=禁用

# 计算总实验数
total_configs=$((${#HORIZON_NORMS[@]} * ${#FUSION_MODES[@]}))
total_pred_lens=${#PRED_LENS[@]}
total_exps=$((total_configs * total_pred_lens))
current=0

echo "=========================================="
echo "T3Time_FreTS_Gated_Qwen_LongSeq 多预测长度训练"
echo "=========================================="
echo "配置数量: ${total_configs}"
echo "  - Horizon归一化: ${HORIZON_NORMS[@]}"
echo "  - 融合模式: ${FUSION_MODES[@]}"
echo "  - 动态稀疏化: ${USE_DYNAMIC_SPARSITY}"
echo "预测长度: ${PRED_LENS[@]}"
echo "总实验数: ${total_exps}"
echo "=========================================="
echo ""

# 遍历配置
for horizon_norm in "${HORIZON_NORMS[@]}"; do
    for fusion_mode in "${FUSION_MODES[@]}"; do
        echo "=========================================="
        echo "配置: horizon_norm=${horizon_norm}, fusion_mode=${fusion_mode}"
        echo "=========================================="
        
        # 遍历预测长度
        for pred_len in "${PRED_LENS[@]}"; do
            current=$((current + 1))
            
            # 生成模型ID
            MODEL_ID="T3Time_FreTS_Gated_Qwen_LongSeq_${horizon_norm}_${fusion_mode}_pred${pred_len}_seed${SEED}"
            
            # 生成日志文件路径
            LOG_FILE="${LOG_DIR}/${horizon_norm}_${fusion_mode}_pred${pred_len}_seed${SEED}.log"
            
            echo "[${current}/${total_exps}] 训练: ${MODEL_ID}"
            echo "  日志: ${LOG_FILE}"
            
            # 运行训练
            python /root/0/T3Time/train_frets_gated_qwen_longseq.py \
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
                --horizon_norm "${horizon_norm}" \
                --fusion_mode "${fusion_mode}" \
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
echo "  python scripts/T3Time_FreTS_FusionExp/analyze_longseq_results.py"
echo "  或直接查看: experiment_results.log"
echo "=========================================="
