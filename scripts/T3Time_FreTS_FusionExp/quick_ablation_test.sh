#!/bin/bash
# 快速消融实验测试脚本（只运行部分关键实验，用于验证脚本是否正常工作）

set -euo pipefail

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate TimeCMA_Qwen3

# 若未设置 PYTHONPATH，用当前项目路径占位
export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_FusionExp_Ablation/ETTh1"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

DATA_PATH="ETTh1"
SEQ_LEN=96
PRED_LEN=96
MODEL_NAME="T3Time_FreTS_Gated_Qwen_FusionExp"
MODEL_ID="T3Time_FreTS_FusionExp_Ablation"
EMBED_VERSION="qwen3_0.6b"

# 基础配置
BASE_CHANNEL=64
BASE_DROPOUT=0.1
BASE_WEIGHT_DECAY=1e-4
BASE_LR=1e-4
BASE_BATCH_SIZE=16
BASE_EPOCHS=10  # 快速测试只用10个epoch
BASE_PATIENCE=5

# 快速测试：只运行几个关键实验
QUICK_TEST_CONFIGS=(
  # 实验1: FreTS Component vs 固定FFT (需要修改模型代码支持)
  "A1_FreTS_Component gate 0.018 0.009 1 1 1 1"
  
  # 实验2: 融合机制对比（只测试2种）
  "A4_Fusion_Gate gate 0.018 0.009 1 1 1 1"
  "A4_Fusion_Weighted weighted 0.018 0.009 1 1 1 1"
  
  # 实验3: 超参数敏感性（只测试3个scale值）
  "A5_Scale_0.015 gate 0.015 0.009 1 1 1 1"
  "A5_Scale_0.018 gate 0.018 0.009 1 1 1 1"
  "A5_Scale_0.020 gate 0.020 0.009 1 1 1 1"
)

append_result() {
  local log_file="$1"
  local exp_name="$2"
  python - <<'PY' "${log_file}" "${exp_name}"
import json, os, re, sys, datetime
log_file = sys.argv[1]
exp_name = sys.argv[2]
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

model_id = os.environ.get("MODEL_ID", "T3Time_FreTS_FusionExp_Ablation")
model_name = os.environ.get("MODEL_NAME", "T3Time_FreTS_Gated_Qwen_FusionExp")

if mse is None or mae is None:
    raise SystemExit(f"[WARN] 未在日志中找到测试指标: {log_file}")

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data = {
    "model_id": f"{model_id}_{exp_name}",
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
    "e_layer": int(os.environ.get("E_LAYER", "1")),
    "d_layer": int(os.environ.get("D_LAYER", "1")),
    "fusion_mode": os.environ["FUSION_MODE"],
    "loss_fn": os.environ.get("LOSS_FN", "smooth_l1"),
    "frets_scale": float(os.environ.get("FRETS_SCALE", "0.02")),
    "sparsity_threshold": float(os.environ.get("SPARSITY_THRESHOLD", "0.01")),
    "ablation_exp": exp_name,
    "lradj": os.environ.get("LRADJ", "type1"),
    "patience": int(os.environ.get("PATIENCE", "10")),
    "gpu": int(os.environ.get("GPU", "0")),
    "embed_version": os.environ["EMBED_VERSION"]
}
print(json.dumps(data, ensure_ascii=False))
PY
}

run_quick_test() {
  local exp_name="$1" fusion_mode="$2" frets_scale="$3" sparsity="$4" \
        use_frets="$5" use_complex="$6" use_sparsity="$7" use_improved_gate="$8"
  
  export DATA_PATH SEQ_LEN MODEL_ID MODEL_NAME EMBED_VERSION
  export PRED_LEN="${PRED_LEN}"
  export LR="${BASE_LR}"
  export CHANNEL="${BASE_CHANNEL}"
  export DROP="${BASE_DROPOUT}"
  export WEIGHT_DECAY="${BASE_WEIGHT_DECAY}"
  export E_LAYER=1
  export D_LAYER=1
  export BATCH_SIZE_RUN="${BASE_BATCH_SIZE}"
  export FUSION_MODE="${fusion_mode}"
  export LOSS_FN="smooth_l1"
  export FRETS_SCALE="${frets_scale}"
  export SPARSITY_THRESHOLD="${sparsity}"
  export LRADJ="type1"
  export PATIENCE="${BASE_PATIENCE}"
  export GPU=0
  export SEED=2024

  log_file="${LOG_DIR}/${exp_name}_seed${SEED}.log"
  echo "=========================================="
  echo "快速测试: ${exp_name}"
  echo "  融合模式: ${fusion_mode}"
  echo "  Scale: ${frets_scale}, Sparsity: ${sparsity}"
  echo "  Epochs: ${BASE_EPOCHS} (快速测试)"
  echo "=========================================="

  cmd=(python /root/0/T3Time/train_frets_gated_qwen_fusion_exp.py
    --data_path "${DATA_PATH}"
    --batch_size "${BATCH_SIZE_RUN}"
    --seq_len "${SEQ_LEN}"
    --pred_len "${PRED_LEN}"
    --epochs "${BASE_EPOCHS}"
    --es_patience "${PATIENCE}"
    --seed "${SEED}"
    --channel "${CHANNEL}"
    --learning_rate "${LR}"
    --dropout_n "${DROP}"
    --weight_decay "${WEIGHT_DECAY}"
    --e_layer "${E_LAYER}"
    --d_layer "${D_LAYER}"
    --fusion_mode "${FUSION_MODE}"
    --loss_fn "${LOSS_FN}"
    --frets_scale "${FRETS_SCALE}"
    --sparsity_threshold "${SPARSITY_THRESHOLD}"
    --lradj "${LRADJ}"
    --embed_version "${EMBED_VERSION}"
    --model_id "${MODEL_ID}_${exp_name}"
  )

  "${cmd[@]}" > "${log_file}" 2>&1

  if [ $? -eq 0 ]; then
    append_result "${log_file}" "${exp_name}" >> "${RESULT_LOG}"
    echo "✅ 完成: ${exp_name}"
  else
    echo "❌ 失败: ${exp_name}, 查看日志: ${log_file}"
  fi
  echo ""
}

echo "=========================================="
echo "快速消融实验测试（用于验证脚本）"
echo "=========================================="
echo "只运行 ${#QUICK_TEST_CONFIGS[@]} 个关键实验"
echo "每个实验只训练 ${BASE_EPOCHS} 个 epoch"
echo "=========================================="
echo ""

total_exps=${#QUICK_TEST_CONFIGS[@]}
current=0

for cfg in "${QUICK_TEST_CONFIGS[@]}"; do
  current=$((current + 1))
  echo "[${current}/${total_exps}] 开始测试..."
  run_quick_test ${cfg}
done

echo "=========================================="
echo "✅ 快速测试完成！"
echo "=========================================="
echo "如果测试成功，可以运行完整消融实验:"
echo "  bash scripts/T3Time_FreTS_FusionExp/ablation_study.sh"
echo ""
echo "查看结果:"
echo "  python scripts/T3Time_FreTS_FusionExp/analyze_ablation_results.py"
