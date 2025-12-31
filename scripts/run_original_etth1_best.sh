#!/bin/bash

# --- 运行环境配置 ---
export CUDA_VISIBLE_DEVICES=1
# 开启显存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 核心实验参数 (基于 ETTh1 最佳配置) ---
DATA_PATH="ETTh1"
MODEL_NAME="T3Time_Original"
SEQ_LEN=96
PRED_LEN=96
CHANNEL=256         # ETTh1 的关键推荐参数
LEARNING_RATE=1e-4
DROPOUT=0.4
SEED=2024
BATCH_SIZE=256      # 原脚本命令行指定的批次大小
EPOCHS=150
E_LAYER=1
D_LAYER=1

# --- 日志路径处理 ---
SAVE_DIR="./logs_benchmark/${DATA_PATH}/"
mkdir -p "$SAVE_DIR"
# 规范化日志文件名：模型_预测长度_通道_种子.log
LOG_FILE="${SAVE_DIR}${MODEL_NAME}_p${PRED_LEN}_c${CHANNEL}_s${SEED}.log"

echo "------------------------------------------------------------"
echo "🚀 启动 T3Time 原版基准测试 (Benchmark) - ETTh1"
echo "数据集: $DATA_PATH | 预测长度: $PRED_LEN"
echo "配置: Channel=$CHANNEL, LR=$LEARNING_RATE, Dropout=$DROPOUT, Seed=$SEED"
echo "------------------------------------------------------------"

# --- 执行训练 ---
# train.py 内部已集成 log_experiment_result，会自动写入 experiment_results.log
nohup python train.py \
    --data_path "$DATA_PATH" \
    --num_nodes 7 \
    --seq_len "$SEQ_LEN" \
    --pred_len "$PRED_LEN" \
    --channel "$CHANNEL" \
    --learning_rate "$LEARNING_RATE" \
    --dropout_n "$DROPOUT" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --e_layer "$E_LAYER" \
    --d_layer "$D_LAYER" \
    --embed_version original \
    > "$LOG_FILE" 2>&1 &

echo "✅ 任务已在后台启动。"
echo "📄 详细日志: $LOG_FILE"
echo "📊 最终结果将自动汇总至: ./experiment_results.log"
echo "------------------------------------------------------------"

