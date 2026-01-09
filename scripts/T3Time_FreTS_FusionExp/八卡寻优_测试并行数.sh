#!/bin/bash
# T3Time_FreTS_Gated_Qwen 并行数测试脚本
# 用于测试每张GPU最多可以同时运行多少个实验
# 只运行少量实验进行测试

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
PARALLEL=${2:-1}  # 并行数（测试参数，默认1）
TEST_EXPS=${3:-10}  # 测试的实验数量（默认10个）

export CUDA_VISIBLE_DEVICES=${GPU_ID}

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_Hyperopt/ETTh1_Test"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

# 固定参数
DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=192
E_LAYER=1
D_LAYER=1
EPOCHS=10  # 测试时减少epochs，加快测试速度
PATIENCE=10
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
MODEL_ID="T3Time_FreTS_Gated_Qwen_Hyperopt_Test"

# 参数搜索空间（使用少量参数组合进行测试）
CHANNELS=(64 96 256)
DROPOUTS=(0.1 0.3 0.5 0.7)
HEADS=(8 16)
LEARNING_RATES=(5e-5 7.5e-5 1e-4)
WEIGHT_DECAYS=(1e-4 5e-4 1e-3 2e-3)
LOSS_FNS=("mse" "smooth_l1")
BATCH_SIZES=(16 32)

# 种子范围：只使用少量种子进行测试
SEEDS=(2000 2001 2002 2003 2004)

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
experiments=()
for seed in "${SEEDS[@]}"; do
    for param_config in "${param_configs[@]}"; do
        experiments+=("${param_config}|${seed}")
    done
done

total_exps=${#experiments[@]}
total_param_configs=${#param_configs[@]}
total_seeds=${#SEEDS[@]}

# 限制测试实验数量
if [ ${total_exps} -gt ${TEST_EXPS} ]; then
    experiments=("${experiments[@]:0:${TEST_EXPS}}")
    total_exps=${TEST_EXPS}
fi

echo "=========================================="
echo "T3Time_FreTS_Gated_Qwen 并行数测试"
echo "GPU: ${GPU_ID}"
echo "并行数: ${PARALLEL}"
echo "测试实验数: ${total_exps}"
echo "Pred_Len: ${PRED_LEN}"
echo "Epochs: ${EPOCHS} (测试模式，已减少)"
echo ""
echo "参数搜索空间:"
echo "  Channel: ${CHANNELS[@]}"
echo "  Dropout: ${DROPOUTS[@]}"
echo "  Head: ${HEADS[@]}"
echo "  Learning Rate: ${LEARNING_RATES[@]}"
echo "  Weight Decay: ${WEIGHT_DECAYS[@]}"
echo "  Loss Function: ${LOSS_FNS[@]}"
echo "  Batch Size: ${BATCH_SIZES[@]}"
echo "种子范围: ${SEEDS[@]} (共 ${total_seeds} 个)"
echo "参数组合数: ${total_param_configs}"
echo "=========================================="
echo ""

# 检查GPU显存
echo "【GPU显存信息】"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | grep "^${GPU_ID}," || echo "无法获取GPU信息"
echo ""

# 运行实验的函数
run_experiment() {
    local exp_idx=$1
    local exp_config=$2
    local gpu_id=$3
    
    IFS='|' read -r CHANNEL DROPOUT_N HEAD LEARNING_RATE WEIGHT_DECAY LOSS_FN BATCH_SIZE SEED <<< "${exp_config}"
    
    log_file="${LOG_DIR}/test_parallel${PARALLEL}_exp${exp_idx}_c${CHANNEL}_d${DROPOUT_N}_h${HEAD}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_loss${LOSS_FN}_bs${BATCH_SIZE}_seed${SEED}.log"
    
    echo "[$(date '+%H:%M:%S')] 实验 ${exp_idx}/${total_exps} 开始 (并行槽位: ${running_jobs}/${PARALLEL})"
    echo "  Channel: ${CHANNEL}, Dropout: ${DROPOUT_N}, Head: ${HEAD}"
    echo "  LR: ${LEARNING_RATE}, WD: ${WEIGHT_DECAY}, Loss: ${LOSS_FN}, BS: ${BATCH_SIZE}, Seed: ${SEED}"
    
    # 记录开始时间和显存
    start_time=$(date +%s)
    start_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${gpu_id} 2>/dev/null || echo "N/A")
    
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
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    end_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${gpu_id} 2>/dev/null || echo "N/A")
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✅ 实验 ${exp_idx} 完成 (耗时: ${duration}秒, 显存: ${start_mem}MB -> ${end_mem}MB)"
    else
        echo "  ❌ 实验 ${exp_idx} 失败 (退出码: ${exit_code}, 耗时: ${duration}秒)"
        echo "     查看日志: ${log_file}"
    fi
}

# 并行运行实验
current_idx=0
running_jobs=0
success_count=0
fail_count=0
start_all_time=$(date +%s)

echo "开始测试并行数 ${PARALLEL}..."
echo ""

while [ ${current_idx} -lt ${total_exps} ]; do
    # 等待有空闲槽位
    while [ ${running_jobs} -ge ${PARALLEL} ]; do
        sleep 2
        # 检查正在运行的作业数
        running_jobs=$(jobs -r | wc -l)
        # 检查是否有失败的作业
        failed_jobs=$(jobs | grep -c "Exit" || echo "0")
    done
    
    # 启动新实验
    exp_config=${experiments[${current_idx}]}
    run_experiment ${current_idx} "${exp_config}" ${GPU_ID} &
    
    current_idx=$((current_idx + 1))
    running_jobs=$(jobs -r | wc -l)
    
    # 每启动一个实验，检查显存使用
    if [ $((current_idx % PARALLEL)) -eq 0 ] || [ ${running_jobs} -eq ${PARALLEL} ]; then
        current_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i ${GPU_ID} 2>/dev/null | awk -F',' '{printf "%.1f%%", ($1/$2)*100}' || echo "N/A")
        echo "  [状态] 运行中: ${running_jobs}/${PARALLEL}, 已完成: $((current_idx - running_jobs)), 显存使用: ${current_mem}"
    fi
done

# 等待所有作业完成
echo ""
echo "等待所有实验完成..."
wait

end_all_time=$(date +%s)
total_duration=$((end_all_time - start_all_time))

# 统计结果
for log in "${LOG_DIR}"/test_parallel${PARALLEL}_*.log; do
    if [ -f "$log" ]; then
        if grep -q "Test MSE:" "$log" 2>/dev/null; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
    fi
done

echo ""
echo "=========================================="
echo "✅ GPU${GPU_ID} 测试完成！"
echo "=========================================="
echo "并行数: ${PARALLEL}"
echo "总实验数: ${total_exps}"
echo "成功: ${success_count}"
echo "失败: ${fail_count}"
echo "总耗时: ${total_duration}秒 ($((${total_duration} / 60))分钟)"
echo "平均每个实验: $((${total_duration} / ${total_exps}))秒"
echo ""
echo "【测试结果】"
if [ ${fail_count} -eq 0 ]; then
    echo "  ✅ 并行数 ${PARALLEL} 测试通过！可以尝试更大的并行数"
else
    echo "  ⚠️  并行数 ${PARALLEL} 有 ${fail_count} 个实验失败"
    echo "     可能原因：显存不足、共享内存不足或其他资源限制"
    echo "     建议：降低并行数或检查日志文件"
fi
echo ""
echo "日志文件保存在: ${LOG_DIR}"
echo "=========================================="
