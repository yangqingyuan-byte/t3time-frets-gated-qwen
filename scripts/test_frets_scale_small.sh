#!/bin/bash
# 测试 FreTS Component 的小 scale 参数
# 测试更小的 scale 值：0.001, 0.002, 0.005

set -euo pipefail

export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "FreTS Component 小 Scale 参数测试"
echo "=========================================="

# 基础参数（对齐 T3Time V30）
DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=96
CHANNEL=128
BATCH_SIZE=16
LEARNING_RATE=1e-4
DROPOUT_N=0.5
WEIGHT_DECAY=1e-3
EPOCHS=100
SEED=2024
FUSION_MODE="gate"
LOSS_FN="mse"
LRADJ="type1"
SPARSITY_THRESHOLD=0.005

# 小 Scale 参数列表（从最小到最大）
SCALES=(0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01)

for scale in "${SCALES[@]}"; do
    echo ""
    echo "=========================================="
    echo "测试 Scale: $scale"
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
        --loss_fn "$LOSS_FN" \
        --lradj "$LRADJ" \
        --sparsity_threshold "$SPARSITY_THRESHOLD" \
        --frets_scale "$scale" \
        --model_id "T3Time_FreTS_FusionExp_scale${scale}"
    
    echo "✅ Scale $scale 实验完成"
done

echo ""
echo "=========================================="
echo "所有小 Scale 参数测试完成！"
echo "=========================================="
echo ""
echo "查看结果："
echo "  python 筛选分析实验结果.py"
echo "  然后选择: T3Time_FreTS_FusionExp_scale"
echo ""
echo "对比所有 scale 值："
echo "  python scripts/analyze_scale_results.py"
echo ""
echo "或者使用命令行："
echo "  grep 'T3Time_FreTS_FusionExp_scale' experiment_results.log | python scripts/analyze_scale_results.py"
