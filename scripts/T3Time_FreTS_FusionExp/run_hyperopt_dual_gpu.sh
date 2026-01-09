#!/bin/bash
# 双GPU并行运行参数寻优的主脚本
# 自动将实验任务分配到两张GPU上

set -uo pipefail

unset __vsc_prompt_cmd_original 2>/dev/null || true

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARALLEL_SCRIPT="${SCRIPT_DIR}/hyperopt_pred720_seed2088_parallel.sh"

# 使用与 hyperopt_pred720_seed2088_parallel.sh 完全相同的逻辑计算总实验数
# 必须按照相同的顺序生成实验配置，确保索引对应关系正确
CHANNELS=(128 256)
DROPOUTS=(0.5 0.6 0.7)
HEADS=(8 16)
LEARNING_RATES=(5e-5 7.5e-5 1e-4)
WEIGHT_DECAYS=(1e-4 5e-4 1e-3 2e-3)
LOSS_FNS=("mse" "smooth_l1")
BATCH_SIZES=(16 32)

# 按照与 hyperopt_pred720_seed2088_parallel.sh 完全相同的顺序生成实验配置
D_LLM=1024
experiments=()
for CHANNEL in "${CHANNELS[@]}"; do
    for DROPOUT_N in "${DROPOUTS[@]}"; do
        for HEAD in "${HEADS[@]}"; do
            # 检查 channel 和 d_llm 是否都能被 head 整除（与 hyperopt_pred720_seed2088_parallel.sh 保持一致）
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

half_exps=$((total_exps / 2))

echo "=========================================="
echo "双GPU并行参数寻优启动脚本"
echo "总实验数: ${total_exps}"
echo "GPU 0: 实验 [0, $((half_exps - 1))]"
echo "GPU 1: 实验 [${half_exps}, $((total_exps - 1))]"
echo "每张GPU并行数: 4"
echo "=========================================="
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARALLEL_SCRIPT="${SCRIPT_DIR}/hyperopt_pred720_seed2088_parallel.sh"

# 在后台启动两个GPU的任务
echo "启动 GPU 0 任务..."
bash "${PARALLEL_SCRIPT}" 0 0 $((half_exps - 1)) 4 > "${SCRIPT_DIR}/gpu0.log" 2>&1 &
GPU0_PID=$!

echo "启动 GPU 1 任务..."
bash "${PARALLEL_SCRIPT}" 1 ${half_exps} $((total_exps - 1)) 4 > "${SCRIPT_DIR}/gpu1.log" 2>&1 &
GPU1_PID=$!

echo ""
echo "两个GPU任务已启动:"
echo "  GPU 0 PID: ${GPU0_PID}"
echo "  GPU 1 PID: ${GPU1_PID}"
echo ""
echo "监控日志:"
echo "  GPU 0: tail -f ${SCRIPT_DIR}/gpu0.log"
echo "  GPU 1: tail -f ${SCRIPT_DIR}/gpu1.log"
echo ""
echo "等待所有任务完成..."

# 等待两个任务完成
wait ${GPU0_PID}
GPU0_EXIT=$?
wait ${GPU1_PID}
GPU1_EXIT=$?

echo ""
echo "=========================================="
if [ ${GPU0_EXIT} -eq 0 ] && [ ${GPU1_EXIT} -eq 0 ]; then
    echo "✅ 所有GPU任务完成！"
else
    echo "⚠️  部分任务可能失败:"
    echo "  GPU 0 退出码: ${GPU0_EXIT}"
    echo "  GPU 1 退出码: ${GPU1_EXIT}"
fi
echo "=========================================="
echo ""
echo "运行分析脚本查看最佳参数组合:"
echo "  python scripts/T3Time_FreTS_FusionExp/find_best_hyperopt_pred720_seed2088.py"
echo "=========================================="
