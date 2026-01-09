#!/bin/bash
# T3Time_FreTS_Gated_Qwen_Hyperopt 并行参数寻优脚本
# 支持多GPU并行运行，可以指定GPU ID和实验范围

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
PARALLEL=${4:-4}  # 默认同时运行4个实验

export CUDA_VISIBLE_DEVICES=${GPU_ID}

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

# 完整搜索空间（基于原版 T3Time 和小波包模型的成功经验）
# 注意：head 必须能整除 channel，所以需要合理搭配
CHANNELS=(128 256)
DROPOUTS=(0.5 0.6 0.7)
# Head 选择：64的因子=[1,2,4,8,16,32,64], 96的因子=[1,2,3,4,6,8,12,16,24,32,48,96], 128的因子=[1,2,4,8,16,32,64,128]
# 选择共同的因子：8, 16，以及96特有的：6, 12
HEADS=(8 16)
LEARNING_RATES=(5e-5 7.5e-5 1e-4)
WEIGHT_DECAYS=(1e-4 5e-4 1e-3 2e-3)
# 注意：bash 数组元素之间用空格分隔，不能用逗号，否则会变成一个字符串 "mse,smooth_l1"
LOSS_FNS=("mse" "smooth_l1")
BATCH_SIZES=(16 32)

# 生成所有实验配置（只包含 channel 和 d_llm 都能被 head 整除的组合）
# 注意：d_llm=1024，所以 head 必须是 1024 的因子（1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024）
experiments=()
D_LLM=1024
for CHANNEL in "${CHANNELS[@]}"; do
    for DROPOUT_N in "${DROPOUTS[@]}"; do
        for HEAD in "${HEADS[@]}"; do
            # 检查 channel 和 d_llm 是否都能被 head 整除
            if [ $((CHANNEL % HEAD)) -eq 0 ] && [ $((D_LLM % HEAD)) -eq 0 ]; then
                for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
                    for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                        for LOSS_FN in "${LOSS_FNS[@]}"; do
                            for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                                experiments+=("${CHANNEL}|${DROPOUT_N}|${HEAD}|${LEARNING_RATE}|${WEIGHT_DECAY}|${LOSS_FN}|${BATCH_SIZE}")
                            done
                        done
                    done
                done
            fi
        done
    done
done

total_exps=${#experiments[@]}

# 如果不带任何参数调用脚本，则默认自动使用两张 GPU 并行跑完所有实验
if [ "$#" -eq 0 ]; then
    MID=$(((total_exps - 1) / 2))
    echo "=========================================="
    echo "检测到未指定参数，启用默认双卡自动并行模式"
    echo "总实验数: ${total_exps}，GPU0 跑 [0, ${MID}]，GPU1 跑 [$((MID + 1)), $((total_exps - 1))]"
    echo "每卡并行数: ${PARALLEL}"
    echo "=========================================="
    echo ""

    # GPU0 跑前半段
    CUDA_VISIBLE_DEVICES=0 bash "$0" 0 0 "${MID}" "${PARALLEL}" &
    PID0=$!

    # GPU1 跑后半段
    CUDA_VISIBLE_DEVICES=1 bash "$0" 1 "$((MID + 1))" "$((total_exps - 1))" "${PARALLEL}" &
    PID1=$!

    # 等待两个子进程结束
    wait "${PID0}" "${PID1}"
    echo "✅ 双卡自动并行超参搜索已完成"
    exit 0
fi

if [ ${END_IDX} -eq -1 ]; then
    END_IDX=$((total_exps - 1))
fi

actual_exps=$((END_IDX - START_IDX + 1))

echo "=========================================="
echo "T3Time_FreTS_Gated_Qwen_Hyperopt 并行参数寻优"
echo "GPU: ${GPU_ID}, 实验范围: [${START_IDX}, ${END_IDX}] / ${total_exps}"
echo "并行数: ${PARALLEL}"
echo "Seed: ${SEED}, Pred_Len: ${PRED_LEN}"
echo "目标: 超越当前最佳 MSE 0.462425，接近原版 T3Time 的 0.438817"
echo ""
echo "搜索空间:"
echo "  Channel: ${CHANNELS[@]}"
echo "  Dropout: ${DROPOUTS[@]}"
echo "  Head: ${HEADS[@]}"
echo "  Learning Rate: ${LEARNING_RATES[@]}"
echo "  Weight Decay: ${WEIGHT_DECAYS[@]}"
echo "  Loss Function: ${LOSS_FNS[@]} (MSE，与原版对齐)"
echo "  Batch Size: ${BATCH_SIZES[@]}"
echo "总实验数: ${total_exps}"
echo "当前GPU将运行: ${actual_exps} 个实验"
echo "=========================================="
echo ""

# 运行实验的函数
run_experiment() {
    local exp_idx=$1
    local exp_config=$2
    local gpu_id=$3
    
    IFS='|' read -r CHANNEL DROPOUT_N HEAD LEARNING_RATE WEIGHT_DECAY LOSS_FN BATCH_SIZE <<< "${exp_config}"
    
    log_file="${LOG_DIR}/pred${PRED_LEN}_c${CHANNEL}_d${DROPOUT_N}_h${HEAD}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_loss${LOSS_FN}_bs${BATCH_SIZE}_seed${SEED}.log"
    
    echo "[实验 ${exp_idx}/${total_exps}] GPU${gpu_id} 开始..."
    echo "  Channel: ${CHANNEL}, Dropout: ${DROPOUT_N}, Head: ${HEAD}"
    echo "  LR: ${LEARNING_RATE}, WD: ${WEIGHT_DECAY}, Loss: ${LOSS_FN}, BS: ${BATCH_SIZE}"
    
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
        echo "  ✅ GPU${gpu_id} 实验 ${exp_idx} 完成"
    else
        echo "  ⚠️  GPU${gpu_id} 实验 ${exp_idx} 失败 (退出码: ${exit_code})"
    fi
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
