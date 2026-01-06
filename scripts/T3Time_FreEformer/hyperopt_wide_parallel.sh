#!/bin/bash
# T3Time_FreEformer_Gated_Qwen 大范围并行参数寻优脚本
# 仿照 T3Time_FreTS_Gated_Qwen_Hyperopt 的风格，在多 GPU 上运行

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
GPU_ID=${1:-0}      # 默认 GPU 0
START_IDX=${2:-0}   # 默认从第0个实验开始
END_IDX=${3:--1}    # 默认到最后一个实验（-1表示全部）
PARALLEL=${4:-4}    # 默认同时运行4个实验

export CUDA_VISIBLE_DEVICES=${GPU_ID}

LOG_DIR="/root/0/T3Time/Results/T3Time_FreEformer/Wide_ETTh1_pred96"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

# 固定参数（基于 Stage1 + Stage2 的经验）
DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=96
NUM_NODES=7
E_LAYER=1
D_LAYER=1
HEAD=8
EPOCHS=80          # 稍短一点，方便大范围搜索
PATIENCE=10
SEED=2021
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
MODEL_ID_PREFIX="T3Time_FreEformer_Wide_ETTh1_pred96"

# 搜索空间（围绕当前最佳点做大范围但可控的搜索）
CHANNELS=(32 48 64)
FRE_E_LAYERS=(1 2)
EMBED_SIZES=(8 16)
LEARNING_RATES=(1e-4 1.25e-4 1.5e-4)
DROPOUTS=(0.4 0.5 0.6)
WEIGHT_DECAYS=(5e-5 1e-4 5e-4)
LOSS_FNS=("smooth_l1")
BATCH_SIZES=(8 16 32)

# 生成所有实验配置
experiments=()
for CHANNEL in "${CHANNELS[@]}"; do
  for FRE_E_LAYER in "${FRE_E_LAYERS[@]}"; do
    for EMBED_SIZE in "${EMBED_SIZES[@]}"; do
      for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
        for DROPOUT_N in "${DROPOUTS[@]}"; do
          for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
            for LOSS_FN in "${LOSS_FNS[@]}"; do
              for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                experiments+=("${CHANNEL}|${FRE_E_LAYER}|${EMBED_SIZE}|${LEARNING_RATE}|${DROPOUT_N}|${WEIGHT_DECAY}|${LOSS_FN}|${BATCH_SIZE}")
              done
            done
          done
        done
      done
    done
  done
done

total_exps=${#experiments[@]}
if [ ${END_IDX} -eq -1 ]; then
  END_IDX=$((total_exps - 1))
fi

actual_exps=$((END_IDX - START_IDX + 1))

echo "=========================================="
echo "T3Time_FreEformer_Gated_Qwen 大范围并行参数寻优"
echo "GPU: ${GPU_ID}, 实验范围: [${START_IDX}, ${END_IDX}] / ${total_exps}"
echo "并行数: ${PARALLEL}"
echo "Seed: ${SEED}, Pred_Len: ${PRED_LEN}"
echo ""
echo "搜索空间:"
echo "  Channel:        ${CHANNELS[@]}"
echo "  Fre_E_Layer:    ${FRE_E_LAYERS[@]}"
echo "  Embed_Size:     ${EMBED_SIZES[@]}"
echo "  Learning Rate:  ${LEARNING_RATES[@]}"
echo "  Dropout:        ${DROPOUTS[@]}"
echo "  Weight Decay:   ${WEIGHT_DECAYS[@]}"
echo "  Loss Function:  ${LOSS_FNS[@]}"
echo "  Batch Size:     ${BATCH_SIZES[@]}"
echo "总实验数: ${total_exps}"
echo "当前GPU将运行: ${actual_exps} 个实验"
echo "=========================================="
echo ""

# 运行实验的函数
run_experiment() {
  local exp_idx=$1
  local exp_config=$2
  local gpu_id=$3

  IFS='|' read -r CHANNEL FRE_E_LAYER EMBED_SIZE LEARNING_RATE DROPOUT_N WEIGHT_DECAY LOSS_FN BATCH_SIZE <<< "${exp_config}"

  local MODEL_ID="${MODEL_ID_PREFIX}_c${CHANNEL}_fre${FRE_E_LAYER}_emb${EMBED_SIZE}_lr${LEARNING_RATE}_d${DROPOUT_N}_wd${WEIGHT_DECAY}_loss${LOSS_FN}_bs${BATCH_SIZE}_seed${SEED}"
  local log_file="${LOG_DIR}/${MODEL_ID}.log"

  echo "[实验 ${exp_idx}/${total_exps}] GPU${gpu_id} 开始..."
  echo "  Channel: ${CHANNEL}, Fre_E_Layer: ${FRE_E_LAYER}, Embed_Size: ${EMBED_SIZE}"
  echo "  LR: ${LEARNING_RATE}, Dropout: ${DROPOUT_N}, WD: ${WEIGHT_DECAY}, Loss: ${LOSS_FN}, BS: ${BATCH_SIZE}"

  python -u /root/0/T3Time/train_freeformer_gated_qwen.py \
    --data_path "${DATA_PATH}" \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${PRED_LEN}" \
    --num_nodes "${NUM_NODES}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --dropout_n "${DROPOUT_N}" \
    --channel "${CHANNEL}" \
    --e_layer "${E_LAYER}" \
    --d_layer "${D_LAYER}" \
    --head "${HEAD}" \
    --epochs "${EPOCHS}" \
    --es_patience "${PATIENCE}" \
    --lradj "${LRADJ}" \
    --embed_version "${EMBED_VERSION}" \
    --seed "${SEED}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --loss_fn "${LOSS_FN}" \
    --model_id "${MODEL_ID}" \
    --embed_size "${EMBED_SIZE}" \
    --fre_e_layer "${FRE_E_LAYER}" \
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

#!/bin/bash
# T3Time_FreEformer_Gated_Qwen 大范围并行参数寻优脚本
# 仿照 T3Time_FreTS_Gated_Qwen_Hyperopt 的风格，在多 GPU 上运行

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
GPU_ID=${1:-0}      # 默认 GPU 0
START_IDX=${2:-0}   # 默认从第0个实验开始
END_IDX=${3:--1}    # 默认到最后一个实验（-1表示全部）
PARALLEL=${4:-4}    # 默认同时运行4个实验

export CUDA_VISIBLE_DEVICES=${GPU_ID}

LOG_DIR="/root/0/T3Time/Results/T3Time_FreEformer/Wide_ETTh1_pred96"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

# 固定参数（基于 Stage1 + Stage2 的经验）
DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=96
NUM_NODES=7
E_LAYER=1
D_LAYER=1
HEAD=8
EPOCHS=80          # 稍短一点，方便大范围搜索
PATIENCE=10
SEED=2021
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
MODEL_ID_PREFIX="T3Time_FreEformer_Wide_ETTh1_pred96"

# 搜索空间（围绕当前最佳点做大范围但可控的搜索）
CHANNELS=(32 48 64)
FRE_E_LAYERS=(1 2)
EMBED_SIZES=(8 16)
LEARNING_RATES=(1e-4 1.25e-4 1.5e-4)
DROPOUTS=(0.4 0.5 0.6)
WEIGHT_DECAYS=(5e-5 1e-4 5e-4)
LOSS_FNS=("smooth_l1")
BATCH_SIZES=(8 16 32)

# 生成所有实验配置
experiments=()
for CHANNEL in "${CHANNELS[@]}"; do
  for FRE_E_LAYER in "${FRE_E_LAYERS[@]}"; do
    for EMBED_SIZE in "${EMBED_SIZES[@]}"; do
      for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
        for DROPOUT_N in "${DROPOUTS[@]}"; do
          for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
            for LOSS_FN in "${LOSS_FNS[@]}"; do
              for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                experiments+=("${CHANNEL}|${FRE_E_LAYER}|${EMBED_SIZE}|${LEARNING_RATE}|${DROPOUT_N}|${WEIGHT_DECAY}|${LOSS_FN}|${BATCH_SIZE}")
              done
            done
          done
        done
      done
    done
  done
done

total_exps=${#experiments[@]}
if [ ${END_IDX} -eq -1 ]; then
  END_IDX=$((total_exps - 1))
fi

actual_exps=$((END_IDX - START_IDX + 1))

echo "=========================================="
echo "T3Time_FreEformer_Gated_Qwen 大范围并行参数寻优"
echo "GPU: ${GPU_ID}, 实验范围: [${START_IDX}, ${END_IDX}] / ${total_exps}"
echo "并行数: ${PARALLEL}"
echo "Seed: ${SEED}, Pred_Len: ${PRED_LEN}"
echo ""
echo "搜索空间:"
echo "  Channel:        ${CHANNELS[@]}"
echo "  Fre_E_Layer:    ${FRE_E_LAYERS[@]}"
echo "  Embed_Size:     ${EMBED_SIZES[@]}"
echo "  Learning Rate:  ${LEARNING_RATES[@]}"
echo "  Dropout:        ${DROPOUTS[@]}"
echo "  Weight Decay:   ${WEIGHT_DECAYS[@]}"
echo "  Loss Function:  ${LOSS_FNS[@]}"
echo "  Batch Size:     ${BATCH_SIZES[@]}"
echo "总实验数: ${total_exps}"
echo "当前GPU将运行: ${actual_exps} 个实验"
echo "=========================================="
echo ""

# 运行实验的函数
run_experiment() {
  local exp_idx=$1
  local exp_config=$2
  local gpu_id=$3

  IFS='|' read -r CHANNEL FRE_E_LAYER EMBED_SIZE LEARNING_RATE DROPOUT_N WEIGHT_DECAY LOSS_FN BATCH_SIZE <<< "${exp_config}"

  local MODEL_ID="${MODEL_ID_PREFIX}_c${CHANNEL}_fre${FRE_E_LAYER}_emb${EMBED_SIZE}_lr${LEARNING_RATE}_d${DROPOUT_N}_wd${WEIGHT_DECAY}_loss${LOSS_FN}_bs${BATCH_SIZE}_seed${SEED}"
  local log_file="${LOG_DIR}/${MODEL_ID}.log"

  echo "[实验 ${exp_idx}/${total_exps}] GPU${gpu_id} 开始..."
  echo "  Channel: ${CHANNEL}, Fre_E_Layer: ${FRE_E_LAYER}, Embed_Size: ${EMBED_SIZE}"
  echo "  LR: ${LEARNING_RATE}, Dropout: ${DROPOUT_N}, WD: ${WEIGHT_DECAY}, Loss: ${LOSS_FN}, BS: ${BATCH_SIZE}"

  python -u /root/0/T3Time/train_freeformer_gated_qwen.py \
    --data_path "${DATA_PATH}" \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${PRED_LEN}" \
    --num_nodes "${NUM_NODES}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --dropout_n "${DROPOUT_N}" \
    --channel "${CHANNEL}" \
    --e_layer "${E_LAYER}" \
    --d_layer "${D_LAYER}" \
    --head "${HEAD}" \
    --epochs "${EPOCHS}" \
    --es_patience "${PATIENCE}" \
    --lradj "${LRADJ}" \
    --embed_version "${EMBED_VERSION}" \
    --seed "${SEED}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --loss_fn "${LOSS_FN}" \
    --model_id "${MODEL_ID}" \
    --embed_size "${EMBED_SIZE}" \
    --fre_e_layer "${FRE_E_LAYER}" \
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
