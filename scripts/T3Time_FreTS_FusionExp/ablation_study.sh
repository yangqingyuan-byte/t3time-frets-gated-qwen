#!/bin/bash
# T3Time_FreTS_FusionExp 消融实验脚本
# 基于论文大纲的消融实验设计

set -uo pipefail  # 移除 -e，允许单个实验失败时继续运行

# 清理可能的环境变量问题
unset __vsc_prompt_cmd_original 2>/dev/null || true

# 激活 conda 环境
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

# 若未设置 PYTHONPATH，用当前项目路径占位
export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

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
BASE_EPOCHS=100
BASE_PATIENCE=10

# 消融实验配置列表
# 格式: "exp_name fusion_mode frets_scale sparsity_threshold use_frets use_complex use_sparsity use_improved_gate"
# use_frets: 1=使用FreTS, 0=使用固定FFT
# use_complex: 1=使用复数, 0=仅幅度 (仅当use_frets=0时有效)
# use_sparsity: 1=使用稀疏化, 0=不使用 (仅当use_frets=1时有效)
# use_improved_gate: 1=改进门控, 0=原始门控

ABLATION_CONFIGS=(
  # ========== 实验1: FreTS Component 的有效性 ==========
  # 1.1 固定 FFT vs FreTS Component
  "A1_FFT_Magnitude gate 0.02 0.01 0 0 0 1"
  "A1_FreTS_Component gate 0.018 0.009 1 1 1 1"
  
  # 1.2 仅幅度 vs 复数信息 (固定FFT情况下)
  "A2_FFT_Magnitude_Only gate 0.02 0.01 0 0 0 1"
  "A2_FFT_Complex gate 0.02 0.01 0 1 0 1"
  
  # 1.3 有无稀疏化机制 (FreTS情况下)
  "A3_FreTS_NoSparsity gate 0.018 0.0 1 1 0 1"
  "A3_FreTS_WithSparsity gate 0.018 0.009 1 1 1 1"
  
  # ========== 实验2: 融合机制对比 ==========
  "A4_Fusion_Gate gate 0.018 0.009 1 1 1 1"
  "A4_Fusion_Weighted weighted 0.018 0.009 1 1 1 1"
  "A4_Fusion_CrossAttn cross_attn 0.018 0.009 1 1 1 1"
  "A4_Fusion_Hybrid hybrid 0.018 0.009 1 1 1 1"
  
  # ========== 实验3: 超参数敏感性分析 ==========
  # 3.1 scale 参数
  "A5_Scale_0.010 gate 0.010 0.009 1 1 1 1"
  "A5_Scale_0.015 gate 0.015 0.009 1 1 1 1"
  "A5_Scale_0.018 gate 0.018 0.009 1 1 1 1"
  "A5_Scale_0.020 gate 0.020 0.009 1 1 1 1"
  "A5_Scale_0.025 gate 0.025 0.009 1 1 1 1"
  
  # 3.2 sparsity_threshold 参数
  "A6_Sparsity_0.005 gate 0.018 0.005 1 1 1 1"
  "A6_Sparsity_0.008 gate 0.018 0.008 1 1 1 1"
  "A6_Sparsity_0.009 gate 0.018 0.009 1 1 1 1"
  "A6_Sparsity_0.012 gate 0.018 0.012 1 1 1 1"
  "A6_Sparsity_0.015 gate 0.018 0.015 1 1 1 1"
  
  # ========== 实验4: 门控机制改进的影响 ==========
  # 注意: 门控机制的改进需要在模型代码中实现，这里假设可以通过参数控制
  "A7_Original_Gate gate 0.018 0.009 1 1 1 0"
  "A7_Improved_Gate gate 0.018 0.009 1 1 1 1"
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
    "use_frets": int(os.environ.get("USE_FRETS", "1")),
    "use_complex": int(os.environ.get("USE_COMPLEX", "1")),
    "use_sparsity": int(os.environ.get("USE_SPARSITY", "1")),
    "lradj": os.environ.get("LRADJ", "type1"),
    "patience": int(os.environ.get("PATIENCE", "10")),
    "gpu": int(os.environ.get("GPU", "0")),
    "embed_version": os.environ["EMBED_VERSION"]
}
print(json.dumps(data, ensure_ascii=False))
PY
}

run_ablation() {
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
  export USE_FRETS="${use_frets}"
  export USE_COMPLEX="${use_complex}"
  export USE_SPARSITY="${use_sparsity}"
  export USE_IMPROVED_GATE="${use_improved_gate}"
  export LRADJ="type1"
  export PATIENCE="${BASE_PATIENCE}"
  export GPU=0

  # 使用固定种子进行消融实验（确保可复现）
  export SEED=2024

  log_file="${LOG_DIR}/${exp_name}_seed${SEED}.log"
  echo "=========================================="
  echo "消融实验: ${exp_name}"
  echo "  融合模式: ${fusion_mode}"
  echo "  Scale: ${frets_scale}, Sparsity: ${sparsity}"
  echo "  Use FreTS: ${use_frets}, Use Complex: ${use_complex}, Use Sparsity: ${use_sparsity}"
  echo "  Log: ${log_file}"
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
    --use_frets "${USE_FRETS}"
    --use_complex "${USE_COMPLEX}"
    --use_sparsity "${USE_SPARSITY}"
    --use_improved_gate "${USE_IMPROVED_GATE}"
    --lradj "${LRADJ}"
    --embed_version "${EMBED_VERSION}"
    --model_id "${MODEL_ID}_${exp_name}"
  )

  "${cmd[@]}" > "${log_file}" 2>&1

  append_result "${log_file}" "${exp_name}" >> "${RESULT_LOG}"
  echo "✅ 完成: ${exp_name}, 结果已写入 ${RESULT_LOG}"
  echo ""
}

# 运行所有消融实验
total_exps=${#ABLATION_CONFIGS[@]}
current=0

for cfg in "${ABLATION_CONFIGS[@]}"; do
  current=$((current + 1))
  echo "[${current}/${total_exps}] 开始消融实验..."
  run_ablation ${cfg}
done

echo "=========================================="
echo "✅ 所有消融实验完成！"
echo "=========================================="
echo "结果已追加到: ${RESULT_LOG}"
echo "日志文件保存在: ${LOG_DIR}"
echo ""
echo "运行分析脚本查看结果:"
echo "  python scripts/T3Time_FreTS_FusionExp/analyze_ablation_results.py"
