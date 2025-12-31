#!/bin/bash
# 实验消歧脚本：找出 FreTS + FreDF 性能不佳的根源
# 对比维度：架构 (FreTS vs FFT) | 损失函数 (MSE vs FreDF)

set -euo pipefail

export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTHONPATH="/root/0/T3Time/plug_and_play:${PYTHONPATH-}"
export CUDA_VISIBLE_DEVICES=0

RESULT_LOG="/root/0/T3Time/ablation_results.log"
DATA_PATH="ETTh1"
PRED_LEN=96
CHANNEL=256
DROPOUT=0.4
BATCH_SIZE=32
EPOCHS=100

echo "开始消歧实验 - $(date)" | tee -a "${RESULT_LOG}"

run_experiment() {
    local model_type="$1"
    local lambda_f="$2"
    local experiment_name="$3"

    for seed in 2024 2025 2026; do
        echo ">>>> 运行实验: ${experiment_name} | Seed: ${seed}"
        
        # 记录到一个临时文件
        log_tmp="ablation_tmp.log"
        
        python train_frets_gated_qwen_fredf.py \
            --model "${model_type}" \
            --lambda_freq "${lambda_f}" \
            --data_path "${DATA_PATH}" \
            --pred_len "${PRED_LEN}" \
            --channel "${CHANNEL}" \
            --dropout_n "${DROPOUT}" \
            --batch_size "${BATCH_SIZE}" \
            --epochs "${EPOCHS}" \
            --seed "${seed}" > "${log_tmp}" 2>&1

        # 提取结果
        # 输出格式为: On average horizons, Test MSE: 0.401030, Test MAE: 0.419029
        mse=$(grep "On average horizons" "${log_tmp}" | awk '{print $6}' | sed 's/,//')
        mae=$(grep "On average horizons" "${log_tmp}" | awk '{print $9}')
        
        # 写入专用消歧日志 (用于 analyze_ablation.py)
        echo "{\"experiment\": \"${experiment_name}\", \"seed\": ${seed}, \"mse\": ${mse}, \"mae\": ${mae}, \"time\": \"$(date '+%Y-%m-%d %H:%M:%S')\"}" >> "${RESULT_LOG}"
        
        # 同时以标准格式写入主实验日志 (用于 筛选分析实验结果.py)
        # 注意：这里 model 名加上了 experiment_name 后缀，方便在主日志中区分
        MAIN_LOG="/root/0/T3Time/experiment_results.log"
        echo "{\"data_path\": \"${DATA_PATH}\", \"pred_len\": ${PRED_LEN}, \"test_mse\": ${mse}, \"test_mae\": ${mae}, \"model\": \"${model_type}_Ablation_${experiment_name}\", \"timestamp\": \"$(date '+%Y-%m-%d %H:%M:%S')\", \"seed\": ${seed}, \"seq_len\": 96, \"channel\": ${CHANNEL}, \"batch_size\": ${BATCH_SIZE}, \"learning_rate\": 0.0001, \"dropout_n\": ${DROPOUT}, \"lambda_freq\": ${lambda_f}}" >> "${MAIN_LOG}"

        echo "结果: MSE=${mse}, MAE=${mae} (已同步至主日志)"
    done
}

# 实验 A: 纯架构测试 (FreTS + MSE)
run_experiment "FreTS_Only" "0.0" "FreTS_Arch_MSE_Loss"

# 实验 B: 当前组合 (FreTS + FreDF)
run_experiment "FreTS_FreDF" "0.5" "FreTS_Arch_FreDF_Loss"

# 实验 C: 传统架构 + FreDF (FFT + FreDF)
run_experiment "FFT_FreDF" "0.5" "FFT_Arch_FreDF_Loss"

# 实验 D: 传统架构基准 (FFT + MSE)
run_experiment "FFT_FreDF" "0.0" "FFT_Arch_MSE_Loss"

echo "消歧实验全部完成，结果已保存至 ${RESULT_LOG}"

