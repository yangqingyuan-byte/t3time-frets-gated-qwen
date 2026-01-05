#!/bin/bash
# FreTS FusionExp 最佳配置训练脚本
# 基于参数寻优结果的最佳配置：
#   - Scale: 0.018
#   - Sparsity Threshold: 0.009
#   - MSE: 0.376336, MAE: 0.390907 (seed 2021)
#
# 说明：
# 1) 使用最佳参数配置，对多个种子进行训练以验证稳定性
# 2) 训练结束后解析日志末尾的测试结果，以 JSONL 追加到 experiment_results.log
# 3) 默认运行种子 2020-2040（21个种子），可自定义

set -euo pipefail

# 若未设置 PYTHONPATH，用当前项目路径占位
export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_FusionExp_Best/ETTh1"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

DATA_PATH="ETTh1"
SEQ_LEN=96
MODEL_NAME="T3Time_FreTS_Gated_Qwen_FusionExp"
MODEL_ID="T3Time_FreTS_FusionExp_Best"
EMBED_VERSION="qwen3_0.6b"
LRADJ="type1"
PATIENCE=10
GPU=0

# 最佳配置参数（基于参数寻优结果）
BEST_SCALE=0.018
BEST_SPARSITY=0.009

# 基础配置（原始最佳配置）
BASE_CHANNEL=64
BASE_DROPOUT=0.1
BASE_WEIGHT_DECAY=1e-4
BASE_LOSS_FN="smooth_l1"
BASE_FUSION_MODE="gate"

# 训练参数
PRED_LEN=96
LR=1e-4
CHANNEL="${BASE_CHANNEL}"
DROP="${BASE_DROPOUT}"
WEIGHT_DECAY="${BASE_WEIGHT_DECAY}"
BATCH_SIZE_RUN=16
FUSION_MODE="${BASE_FUSION_MODE}"
LOSS_FN="${BASE_LOSS_FN}"
FRETS_SCALE="${BEST_SCALE}"
SPARSITY_THRESHOLD="${BEST_SPARSITY}"

# 种子列表（默认 2020-2040，可自定义）
SEEDS="${1:-2020 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040}"

append_result() {
  local log_file="$1"
  python - <<'PY' "${log_file}"
import json, os, re, sys, datetime
log_file = sys.argv[1]
text = open(log_file, 'r', encoding='utf-8', errors='ignore').read().splitlines()

mse = mae = None
for line in reversed(text):
    if "Test MSE" in line or "On average horizons" in line:
        m = re.search(r"MSE:\s*([0-9.]+)", line)
        if m:
            mse = float(m.group(1))
            m2 = re.search(r"MAE:\s*([0-9.]+)", line)
            if m2:
                mae = float(m2.group(1))
            break

model_id = os.environ.get("MODEL_ID", "T3Time_FreTS_FusionExp_Best")
model_name = os.environ.get("MODEL_NAME", "T3Time_FreTS_Gated_Qwen_FusionExp")

if mse is None or mae is None:
    raise SystemExit(f"[WARN] 未在日志中找到测试指标: {log_file}")

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data = {
    "model_id": model_id, # 确保 model_id 在第一个位置
    "data_path": os.environ["DATA_PATH"],
    "pred_len": int(os.environ["PRED_LEN"]),
    "test_mse": mse,
    "test_mae": mae,
    "model": model_name,
    "timestamp": now,
    "seed": int(os.environ["SEED"]),
    "seq_len": int(os.environ["SEQ_LEN"]),
    "channel": int(os.environ["CHANNEL"]),
    "batch_size": int(os.environ["BATCH_SIZE_RUN"]),
    "learning_rate": float(os.environ["LR"]),
    "dropout_n": float(os.environ["DROP"]),
    "weight_decay": float(os.environ["WEIGHT_DECAY"]),
    "fusion_mode": os.environ["FUSION_MODE"],
    "loss_fn": os.environ["LOSS_FN"],
    "frets_scale": float(os.environ["FRETS_SCALE"]),
    "sparsity_threshold": float(os.environ["SPARSITY_THRESHOLD"]),
    "lradj": os.environ["LRADJ"],
    "patience": int(os.environ["PATIENCE"]),
    "gpu": int(os.environ["GPU"]),
    "embed_version": os.environ["EMBED_VERSION"]
}
print(json.dumps(data, ensure_ascii=False))
PY
}

echo "=========================================="
echo "FreTS FusionExp 最佳配置训练"
echo "=========================================="
echo "配置参数:"
echo "  Scale: ${FRETS_SCALE}"
echo "  Sparsity Threshold: ${SPARSITY_THRESHOLD}"
echo "  Channel: ${CHANNEL}"
echo "  Dropout: ${DROP}"
echo "  Weight Decay: ${WEIGHT_DECAY}"
echo "  Loss Function: ${LOSS_FN}"
echo "  Fusion Mode: ${FUSION_MODE}"
echo "  Seeds: ${SEEDS}"
echo "=========================================="
echo ""

export DATA_PATH SEQ_LEN MODEL_ID MODEL_NAME EMBED_VERSION LRADJ PATIENCE GPU
export PRED_LEN LR CHANNEL DROP WEIGHT_DECAY BATCH_SIZE_RUN FUSION_MODE LOSS_FN FRETS_SCALE SPARSITY_THRESHOLD

total_seeds=$(echo ${SEEDS} | wc -w)
current=0

for seed in ${SEEDS}; do
  current=$((current + 1))
  export SEED="${seed}"

  log_file="${LOG_DIR}/best_scale${FRETS_SCALE}_sparsity${SPARSITY_THRESHOLD}_seed${seed}.log"
  echo "[${current}/${total_seeds}] 开始训练: seed=${seed}, log=${log_file}"

  cmd=(python /root/0/T3Time/train_frets_gated_qwen_fusion_exp.py
    --data_path "${DATA_PATH}"
    --pred_len "${PRED_LEN}"
    --channel "${CHANNEL}"
    --batch_size "${BATCH_SIZE_RUN}"
    --learning_rate "${LR}"
    --dropout_n "${DROP}"
    --weight_decay "${WEIGHT_DECAY}"
    --fusion_mode "${FUSION_MODE}"
    --loss_fn "${LOSS_FN}"
    --frets_scale "${FRETS_SCALE}"
    --sparsity_threshold "${SPARSITY_THRESHOLD}"
    --lradj "${LRADJ}"
    --epochs 100
    --es_patience "${PATIENCE}"
    --seed "${SEED}"
    --embed_version "${EMBED_VERSION}"
    --model_id "${MODEL_ID}"
  )

  "${cmd[@]}" > "${log_file}" 2>&1

  append_result "${log_file}" >> "${RESULT_LOG}"
  echo "  ✅ 已完成，结果已写入 ${RESULT_LOG}"
  echo ""
done

echo "=========================================="
echo "✅ 所有训练完成！"
echo "=========================================="
echo "结果已追加到: ${RESULT_LOG}"
echo "日志文件保存在: ${LOG_DIR}"
echo ""
echo "查看结果:"
echo "  grep '${MODEL_ID}' ${RESULT_LOG} | python -c \""
echo "import sys, json"
echo "results = []"
echo "for line in sys.stdin:"
echo "    data = json.loads(line.strip())"
echo "    results.append((data['seed'], data['test_mse'], data['test_mae']))"
echo "results.sort(key=lambda x: x[1])"
echo "print('最佳结果 (按 MSE 排序):')"
echo "for seed, mse, mae in results[:5]:"
echo "    print(f'  Seed {seed}: MSE={mse:.6f}, MAE={mae:.6f}')"
echo "avg_mse = sum(r[1] for r in results) / len(results)"
echo "avg_mae = sum(r[2] for r in results) / len(results)"
echo "print(f'\\n平均结果: MSE={avg_mse:.6f}, MAE={avg_mae:.6f}')"
echo "\""
