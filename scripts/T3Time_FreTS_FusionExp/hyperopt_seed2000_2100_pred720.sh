#!/bin/bash
# T3Time_FreTS_Gated_Qwen 多种子寻优脚本 (Seed 2000-2100)
# 固定最佳配置：Channel=64, Dropout=0.5, Head=8, Pred_Len=720
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
PARALLEL=${4:-4}  # 默认同时运行4个实验

export CUDA_VISIBLE_DEVICES=${GPU_ID}

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_Hyperopt/ETTh1"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

# 固定参数（基于最佳配置）
DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=720
CHANNEL=64
DROPOUT_N=0.5
HEAD=8
E_LAYER=1
D_LAYER=1
EPOCHS=150
PATIENCE=10
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
MODEL_ID="T3Time_FreTS_Gated_Qwen_Hyperopt"

# 根据最佳配置设置的其他参数
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
LOSS_FN="mse"
BATCH_SIZE=16

# 种子范围：2000-2100（共101个种子）
SEEDS=()
for seed in $(seq 2000 2100); do
    SEEDS+=(${seed})
done

total_exps=${#SEEDS[@]}

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
    echo "✅ 双卡自动并行多种子搜索已完成"
    exit 0
fi

if [ ${END_IDX} -eq -1 ]; then
    END_IDX=$((total_exps - 1))
fi

actual_exps=$((END_IDX - START_IDX + 1))

echo "=========================================="
echo "T3Time_FreTS_Gated_Qwen 多种子寻优 (Seed 2000-2100)"
echo "GPU: ${GPU_ID}, 实验范围: [${START_IDX}, ${END_IDX}] / ${total_exps}"
echo "并行数: ${PARALLEL}"
echo "Pred_Len: ${PRED_LEN}"
echo ""
echo "固定配置:"
echo "  Channel: ${CHANNEL}"
echo "  Dropout: ${DROPOUT_N}"
echo "  Head: ${HEAD}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Weight Decay: ${WEIGHT_DECAY}"
echo "  Loss Function: ${LOSS_FN}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  E_Layer: ${E_LAYER}"
echo "  D_Layer: ${D_LAYER}"
echo "总种子数: ${total_exps}"
echo "当前GPU将运行: ${actual_exps} 个实验"
echo "=========================================="
echo ""

# 运行实验的函数
run_experiment() {
    local exp_idx=$1
    local seed=$2
    local gpu_id=$3
    
    log_file="${LOG_DIR}/pred${PRED_LEN}_c${CHANNEL}_d${DROPOUT_N}_h${HEAD}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_loss${LOSS_FN}_bs${BATCH_SIZE}_seed${seed}.log"
    
    echo "[实验 ${exp_idx}/${total_exps}] GPU${gpu_id} 开始..."
    echo "  Seed: ${seed}"
    
    python /root/0/T3Time/train_frets_gated_qwen.py \
        --data_path "${DATA_PATH}" \
        --batch_size "${BATCH_SIZE}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${PRED_LEN}" \
        --epochs "${EPOCHS}" \
        --es_patience "${PATIENCE}" \
        --seed "${seed}" \
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
        echo "  ✅ GPU${gpu_id} 实验 ${exp_idx} (Seed ${seed}) 完成"
    else
        echo "  ⚠️  GPU${gpu_id} 实验 ${exp_idx} (Seed ${seed}) 失败 (退出码: ${exit_code})"
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
    seed=${SEEDS[${current_idx}]}
    run_experiment ${current_idx} "${seed}" ${GPU_ID} &
    
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
