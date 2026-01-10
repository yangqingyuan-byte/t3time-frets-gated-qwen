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

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_Hyperopt/ETTm2"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

# 固定参数
DATA_PATH="ETTm2"
SEQ_LEN=96
PRED_LEN=192
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
    echo "种子数: ${total_seeds} (${SEEDS[0]}-${SEEDS[-1]})"
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
echo "Pred_Len: ${PRED_LEN}"
echo ""
echo "参数搜索空间:"
echo "  Channel: ${CHANNELS[@]}"
echo "  Dropout: ${DROPOUTS[@]}"
echo "  Head: ${HEADS[@]}"
echo "  Learning Rate: ${LEARNING_RATES[@]}"
echo "  Weight Decay: ${WEIGHT_DECAYS[@]}"
echo "  Loss Function: ${LOSS_FNS[@]}"
echo "  Batch Size: ${BATCH_SIZES[@]}"
echo "种子范围: ${SEEDS[0]}-${SEEDS[-1]} (共 ${total_seeds} 个)"
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
    
    log_file="${LOG_DIR}/pred${PRED_LEN}_c${CHANNEL}_d${DROPOUT_N}_h${HEAD}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_loss${LOSS_FN}_bs${BATCH_SIZE}_seed${SEED}.log"
    
    # 不在这里打印，避免干扰进度显示
    # 信息会记录在日志文件中
    
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
    # 结果记录在日志中，不在这里打印以避免干扰进度显示
}

# 并行运行实验（带进度跟踪）
current_idx=${START_IDX}
running_jobs=0
start_time=$(date +%s)
progress_file="/tmp/t3time_progress_gpu${GPU_ID}.txt"
echo "0" > "$progress_file"
exp_durations_file="/tmp/t3time_durations_gpu${GPU_ID}.txt"
> "$exp_durations_file"

# 进度显示函数
show_progress() {
    local current_time=$(date +%s)
    local elapsed=$((current_time - start_time))
    
    # 统计已完成的实验（通过检查日志文件）
    local completed=0
    if [ -d "${LOG_DIR}" ]; then
        # 统计包含 "Test MSE" 或 "Test MAE" 的日志文件数量（表示实验已完成）
        completed=$(grep -l "Test MSE\|Test MAE" "${LOG_DIR}"/*.log 2>/dev/null | wc -l)
    fi
    
    # 读取已启动的实验数（从进度文件）
    local started=$(cat "$progress_file" 2>/dev/null || echo "0")
    local remaining=$((actual_exps - completed))
    
    # 计算平均每个实验的用时
    local avg_time=0
    if [ -f "$exp_durations_file" ] && [ -s "$exp_durations_file" ]; then
        local count=$(wc -l < "$exp_durations_file")
        if [ $count -gt 0 ]; then
            local total_duration=$(awk '{sum+=$1} END {print sum}' "$exp_durations_file")
            avg_time=$((total_duration / count))
        fi
    fi
    
    # 预估剩余时间（基于平均用时和剩余实验数）
    local estimated_remaining=0
    if [ $avg_time -gt 0 ] && [ $remaining -gt 0 ] && [ $running_jobs -gt 0 ]; then
        # 考虑并行度，剩余时间 = (剩余实验数 / 并行度) * 平均用时
        estimated_remaining=$(( (remaining * avg_time) / running_jobs ))
    fi
    
    # 格式化时间
    local elapsed_str=$(printf "%02d:%02d:%02d" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))
    local estimated_str="计算中..."
    if [ $estimated_remaining -gt 0 ]; then
        estimated_str=$(printf "%02d:%02d:%02d" $((estimated_remaining/3600)) $((estimated_remaining%3600/60)) $((estimated_remaining%60)))
    fi
    
    # 计算进度百分比
    local progress_pct=0
    if [ $actual_exps -gt 0 ]; then
        progress_pct=$(awk "BEGIN {printf \"%.1f\", ($completed * 100.0) / $actual_exps}")
    fi
    
    # 清空当前行并显示进度
    printf "\r[进度] 已完成: %d/%d (%.1f%%) | 剩余: %d | 已用时间: %s | 预估剩余: %s | 运行中: %d/%d" \
        "$completed" "$actual_exps" "$progress_pct" "$remaining" "$elapsed_str" "$estimated_str" "$running_jobs" "$PARALLEL"
}

while [ ${current_idx} -le ${END_IDX} ]; do
    # 等待有空闲槽位
    while [ ${running_jobs} -ge ${PARALLEL} ]; do
        sleep 5
        # 检查正在运行的作业数
        running_jobs=$(jobs -r | wc -l)
        # 更新进度显示
        show_progress
    done
    
    # 启动新实验（记录开始时间）
    exp_config=${experiments[${current_idx}]}
    exp_start_time=$(date +%s)
    run_experiment ${current_idx} "${exp_config}" ${GPU_ID} &
    local job_pid=$!
    
    current_idx=$((current_idx + 1))
    echo "$current_idx" > "$progress_file"
    running_jobs=$(jobs -r | wc -l)
    
    # 在后台跟踪实验完成时间
    (
        wait $job_pid
        exp_end_time=$(date +%s)
        exp_duration=$((exp_end_time - exp_start_time))
        echo "$exp_duration" >> "$exp_durations_file"
        # 只保留最近100个实验的用时记录
        if [ $(wc -l < "$exp_durations_file") -gt 100 ]; then
            tail -n 100 "$exp_durations_file" > "${exp_durations_file}.tmp"
            mv "${exp_durations_file}.tmp" "$exp_durations_file"
        fi
    ) &
    
    # 更新进度显示
    show_progress
done

# 等待所有作业完成
echo ""
echo ""
echo "等待所有实验完成..."

# 在等待期间持续更新进度显示
while [ $(jobs -r | wc -l) -gt 0 ]; do
    sleep 5
    running_jobs=$(jobs -r | wc -l)
    show_progress
done

wait

# 最终进度显示
final_time=$(date +%s)
total_elapsed=$((final_time - start_time))
total_elapsed_str=$(printf "%02d:%02d:%02d" $((total_elapsed/3600)) $((total_elapsed%3600/60)) $((total_elapsed%60)))

# 最终统计完成的实验数
final_completed=0
if [ -d "${LOG_DIR}" ]; then
    final_completed=$(grep -l "Test MSE\|Test MAE" "${LOG_DIR}"/*.log 2>/dev/null | wc -l)
fi

echo ""
echo "=========================================="
echo "✅ GPU${GPU_ID} 所有实验完成！"
echo "已完成: ${final_completed}/${actual_exps} 个实验"
echo "总用时: ${total_elapsed_str}"
echo "=========================================="

# 清理临时文件
rm -f "$progress_file" "$exp_durations_file"
echo "结果已追加到: ${RESULT_LOG}"
echo "日志文件保存在: ${LOG_DIR}"
echo ""
