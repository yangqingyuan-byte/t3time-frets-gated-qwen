#!/bin/bash
# 查找多种子寻优实验中的最佳种子（MSE和MAE）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_FILE="${1:-/root/0/T3Time/experiment_results.log}"

echo "=========================================="
echo "查找最佳种子（MSE和MAE）"
echo "=========================================="
echo "结果文件: $RESULT_FILE"
echo ""

python3 "$SCRIPT_DIR/analyze_seed2000_2100_pred720.py" \
    --result_file "$RESULT_FILE" \
    --seed_start 2000 \
    --seed_end 2100 \
    --pred_len 720 \
    --model_id "T3Time_FreTS_Gated_Qwen_Hyperopt"
