#!/bin/bash
# 融合机制对比实验脚本
# 测试 4 种不同的融合方式

set -euo pipefail

export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "开始融合机制对比实验"
echo "=========================================="

# 基础参数
DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=96
CHANNEL=64
BATCH_SIZE=16
LEARNING_RATE=1e-4
DROPOUT_N=0.1
WEIGHT_DECAY=1e-4
EPOCHS=100
SEED=2024

# 融合模式列表
FUSION_MODES=("gate" "weighted" "cross_attn" "hybrid")
FUSION_NAMES=("Gate" "Weighted" "CrossAttn" "Hybrid")

for i in "${!FUSION_MODES[@]}"; do
    FUSION_MODE="${FUSION_MODES[$i]}"
    FUSION_NAME="${FUSION_NAMES[$i]}"
    
    echo ""
    echo "=========================================="
    echo "实验 $((i+1))/4: $FUSION_NAME 融合 ($FUSION_MODE)"
    echo "=========================================="
    
    python train_frets_gated_qwen_fusion_exp.py \
        --data_path "$DATA_PATH" \
        --seq_len "$SEQ_LEN" \
        --pred_len "$PRED_LEN" \
        --channel "$CHANNEL" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --dropout_n "$DROPOUT_N" \
        --weight_decay "$WEIGHT_DECAY" \
        --epochs "$EPOCHS" \
        --seed "$SEED" \
        --fusion_mode "$FUSION_MODE" \
        --model_id "T3Time_FreTS_FusionExp"
    
    echo "✅ $FUSION_NAME 融合实验完成"
done

echo ""
echo "=========================================="
echo "所有融合机制对比实验完成！"
echo "=========================================="
echo ""
echo "查看结果："
echo "  python 筛选分析实验结果.py"
echo "  然后选择: T3Time_FreTS_Gated_Qwen_FusionExp"
