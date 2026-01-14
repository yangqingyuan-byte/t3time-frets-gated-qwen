#!/bin/bash
# T3Time_FreTS_Gated_Qwen 超参数+多种子寻优脚本
# 在指定参数空间搜索，并对每个参数组合在种子 2000-2100 范围内测试
# 支持多GPU并行运行

set -uo pipefail

# 清理可能的环境变量问题
unset __vsc_prompt_cmd_original 2>/dev/null || true

# 激活 conda 环境
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

# 设置环境变量
export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 参数解析
GPU_ID=${1:-0}  # 默认 GPU 0
START_IDX=${2:-0}  # 默认从第0个实验开始
END_IDX=${3:--1}  # 默认到最后一个实验（-1表示全部）
PARALLEL=${4:-5}  # 默认同时运行5个实验

export CUDA_VISIBLE_DEVICES=${GPU_ID}

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_Hyperopt/ETTh2"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

# 固定参数
DATA_PATH="ETTh2"
SEQ_LEN=96
# 支持多个预测长度，依次运行
PRED_LENS=(96 336 720)
E_LAYER=1
D_LAYER=1
EPOCHS=150
PATIENCE=10
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
MODEL_ID="T3Time_FreTS_Gated_Qwen_Hyperopt"

# 参数搜索空间
CHANNELS=(64 96 256)
DROPOUTS=(0.1 0.3 0.5 0.7)
HEADS=(8 16)
LEARNING_RATES=(5e-5 7.5e-5 1e-4)
WEIGHT_DECAYS=(1e-4 5e-4 1e-3 2e-3)
LOSS_FNS=("mse" "smooth_l1")
BATCH_SIZES=(16 32)

# 种子范围：2025-2026（共2个种子）
SEEDS=()
for seed in $(seq 2025 2026); do
    SEEDS+=(${seed})
done

# 生成所有参数组合（只包含 channel 和 d_llm 都能被 head 整除的组合）
D_LLM=1024
param_configs=()
for CHANNEL in "${CHANNELS[@]}"; do
    for DROPOUT_N in "${DROPOUTS[@]}"; do
        for HEAD in "${HEADS[@]}"; do
            # 检查 channel 和 d_llm 是否都能被 head 整除
            if [ $((CHANNEL % HEAD)) -eq 0 ] && [ $((D_LLM % HEAD)) -eq 0 ]; then
                for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
                    for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                        for LOSS_FN in "${LOSS_FNS[@]}"; do
                            for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                                param_configs+=("${CHANNEL}|${DROPOUT_N}|${HEAD}|${LEARNING_RATE}|${WEIGHT_DECAY}|${LOSS_FN}|${BATCH_SIZE}")
                            done
                        done
                    done
                done
            fi
        done
    done
done

# 生成所有实验配置（每个种子 × 每个参数组合）
# 优先搜索参数：在相同种子上跑不同参数
experiments=()
for seed in "${SEEDS[@]}"; do
    for param_config in "${param_configs[@]}"; do
        experiments+=("${param_config}|${seed}")
    done
done

total_exps=${#experiments[@]}
total_param_configs=${#param_configs[@]}
total_seeds=${#SEEDS[@]}

# 如果不带任何参数调用脚本，则默认自动使用八张 GPU 并行跑完所有实验
if [ "$#" -eq 0 ]; then
    NUM_GPUS=8
    EXP_PER_GPU=$((total_exps / NUM_GPUS))
    REMAINDER=$((total_exps % NUM_GPUS))
    
    echo "=========================================="
    echo "检测到未指定参数，启用默认八卡自动并行模式"
    echo "参数组合数: ${total_param_configs}"
    echo "种子数: ${total_seeds} (2000-2100)"
    echo "总实验数: ${total_exps} (${total_param_configs} × ${total_seeds})"
    echo "GPU数量: ${NUM_GPUS}"
    echo "每卡并行数: ${PARALLEL}"
    echo "实验分配:"
    
    # 计算每张GPU的实验范围并启动进程
    PIDS=()
    start_idx=0
    
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        # 前 REMAINDER 张GPU多分配一个实验
        if [ ${gpu} -lt ${REMAINDER} ]; then
            end_idx=$((start_idx + EXP_PER_GPU))
        else
            end_idx=$((start_idx + EXP_PER_GPU - 1))
        fi
        
        # 确保最后一张GPU包含所有剩余实验
        if [ ${gpu} -eq $((NUM_GPUS - 1)) ]; then
            end_idx=$((total_exps - 1))
        fi
        
        echo "  GPU${gpu}: 实验 [$start_idx, $end_idx] ($((end_idx - start_idx + 1)) 个)"
        
        # 启动子进程
        CUDA_VISIBLE_DEVICES=${gpu} bash "$0" ${gpu} ${start_idx} ${end_idx} "${PARALLEL}" &
        PIDS+=($!)
        
        start_idx=$((end_idx + 1))
    done
    
    echo "=========================================="
    echo ""
    
    # 等待所有8个子进程结束
    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done
    
    echo "✅ 八卡自动并行超参数+多种子搜索已完成"
    exit 0
fi

if [ ${END_IDX} -eq -1 ]; then
    END_IDX=$((total_exps - 1))
fi

actual_exps=$((END_IDX - START_IDX + 1))

echo "=========================================="
echo "T3Time_FreTS_Gated_Qwen 超参数+多种子寻优"
echo "GPU: ${GPU_ID}, 实验范围: [${START_IDX}, ${END_IDX}] / ${total_exps}"
echo "并行数: ${PARALLEL}"
echo "Pred_Lens: ${PRED_LENS[@]}"
echo ""
echo "参数搜索空间:"
echo "  Channel: ${CHANNELS[@]}"
echo "  Dropout: ${DROPOUTS[@]}"
echo "  Head: ${HEADS[@]}"
echo "  Learning Rate: ${LEARNING_RATES[@]}"
echo "  Weight Decay: ${WEIGHT_DECAYS[@]}"
echo "  Loss Function: ${LOSS_FNS[@]}"
echo "  Batch Size: ${BATCH_SIZES[@]}"
echo "种子范围: 2000-2100 (共 ${total_seeds} 个)"
echo "参数组合数: ${total_param_configs}"
echo "总实验数: ${total_exps} (${total_param_configs} × ${total_seeds})"
echo "当前GPU将运行: ${actual_exps} 个实验"
echo "=========================================="
echo ""

# 运行实验的函数
run_experiment() {
    local exp_idx=$1
    local exp_config=$2
    local gpu_id=$3
    
    IFS='|' read -r CHANNEL DROPOUT_N HEAD LEARNING_RATE WEIGHT_DECAY LOSS_FN BATCH_SIZE SEED <<< "${exp_config}"
    
    echo "[实验 ${exp_idx}/${total_exps}] GPU${gpu_id} 开始..."
    echo "  Channel: ${CHANNEL}, Dropout: ${DROPOUT_N}, Head: ${HEAD}"
    echo "  LR: ${LEARNING_RATE}, WD: ${WEIGHT_DECAY}, Loss: ${LOSS_FN}, BS: ${BATCH_SIZE}"
    echo "  Seed: ${SEED}"
    echo "  将依次运行预测长度: ${PRED_LENS[@]}"

    # 依次在多个预测长度上运行
    for PRED_LEN in "${PRED_LENS[@]}"; do
        log_file="${LOG_DIR}/pred${PRED_LEN}_c${CHANNEL}_d${DROPOUT_N}_h${HEAD}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_loss${LOSS_FN}_bs${BATCH_SIZE}_seed${SEED}.log"
        echo "    -> 开始 Pred_Len=${PRED_LEN} ..."
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
            echo "    ✅ Pred_Len=${PRED_LEN} 完成"
        else
            echo "    ⚠️ Pred_Len=${PRED_LEN} 失败 (退出码: ${exit_code})"
        fi
    done

    echo "  ✅ GPU${gpu_id} 实验 ${exp_idx} 的所有预测长度已运行完毕"
}

# 并行运行实验
current_idx=${START_IDX}
running_jobs=0

while [ ${current_idx} -le ${END_IDX} ]; do
    # 等待有空闲槽位
    while [ ${running_jobs} -ge ${PARALLEL} ]; do
        sleep 5
        # 检查正在运行的作业数
        running_jobs=$(jobs -r | wc -l)
    done
    
    # 启动新实验
    exp_config=${experiments[${current_idx}]}
    run_experiment ${current_idx} "${exp_config}" ${GPU_ID} &
    
    current_idx=$((current_idx + 1))
    running_jobs=$(jobs -r | wc -l)
    
    echo "  当前运行中: ${running_jobs}/${PARALLEL} 个实验"
done

# 等待所有作业完成
echo ""
echo "等待所有实验完成..."
wait

echo "=========================================="
echo "✅ GPU${GPU_ID} 所有实验完成！"
echo "=========================================="
echo "结果已追加到: ${RESULT_LOG}"
echo "日志文件保存在: ${LOG_DIR}"
echo ""

# 发送微信通知（如果配置了 SENDKEY 环境变量）
if [ -n "${SENDKEY:-}" ] || [ -n "${QYWX_CORPID:-}" ]; then
    echo "正在发送完成通知..."
    python /root/0/T3Time/notify_wechat.py \
        --title "ETTh2 八卡寻优完成 (GPU${GPU_ID})" \
        --body "✅ 所有实验已完成

数据集: ETTh2
GPU: ${GPU_ID}
实验范围: [${START_IDX}, ${END_IDX}] / ${total_exps}
完成时间: $(date '+%Y-%m-%d %H:%M:%S')
日志目录: ${LOG_DIR}" || echo "通知发送失败（不影响实验结果）"
fi
