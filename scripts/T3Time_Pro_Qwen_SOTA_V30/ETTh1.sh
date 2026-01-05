#!/bin/bash
# 说明：
# 1) 针对 T3Time_Pro_Qwen_SOTA_V30 模型做参数寻优，每个配置对 seed 2020-2040 逐一运行；
# 2) 训练结束后解析日志末尾的 "Pro Test Results: MSE/MAE"，以 JSONL 追加到 /root/0/T3Time/experiment_results.log；
# 3) 默认顺序运行，确保记录稳定。若需并行可自行在外层加 nohup &。

set -euo pipefail

# 若未设置 PYTHONPATH，用当前项目路径占位
export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

LOG_DIR="/root/0/T3Time/Results/T3Time_Pro_Qwen_SOTA_V30/ETTh1"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

DATA_PATH="ETTh1"
SEQ_LEN=96
MODEL_NAME="TriModalLearnableWaveletPacketGatedProQwen"
MODEL_ID="T3Time_Pro_Qwen_SOTA_V30"
EMBED_VERSION="qwen3_0.6b"
WP_LEVEL=2
LRADJ="type1"
PATIENCE=15
GPU=0

# 配置列表：
# pred_len lr channel dropout_n weight_decay batch_size epochs
# 可按需增删组合以扩大/缩小搜索空间
CONFIGS=(
  # pred_len 96 - 基础配置
  "96 1e-4 128 0.3 1e-3 16 100"
  "96 1e-4 128 0.4 1e-3 16 100"
  "96 1e-4 128 0.5 1e-3 16 100"
  "96 5e-5 128 0.5 1e-3 16 100"
  "96 1e-4 256 0.5 1e-3 16 100"
  
  # pred_len 192
  "192 1e-4 128 0.5 1e-3 16 100"
  "192 5e-5 128 0.5 1e-3 16 100"
  "192 1e-4 256 0.5 1e-3 16 100"
  
  # pred_len 336
  "336 1e-4 128 0.5 1e-3 16 100"
  "336 5e-5 128 0.6 1e-3 16 100"
  
  # pred_len 720
  "720 1e-4 128 0.5 1e-3 16 100"
  "720 5e-5 128 0.6 1e-3 16 100"
)

append_result() {
  local log_file="$1"
  python - <<'PY' "${log_file}"
import json, os, re, sys, datetime
log_file = sys.argv[1]
text = open(log_file, 'r', encoding='utf-8', errors='ignore').read().splitlines()

mse = mae = None
# V30 训练脚本输出格式: "Pro Test Results: MSE: 0.123456, MAE: 0.123456"
for line in reversed(text):
    if "Pro Test Results" in line:
        m = re.search(r"MSE:\s*([0-9.]+),\s*MAE:\s*([0-9.]+)", line)
        if m:
            mse, mae = float(m.group(1)), float(m.group(2))
            break

if mse is None or mae is None:
    raise SystemExit(f"[WARN] 未在日志中找到测试指标: {log_file}")

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data = {
    "model_id": os.environ.get("MODEL_ID", "T3Time_Pro_Qwen_SOTA_V30"),
    "data_path": os.environ["DATA_PATH"],
    "pred_len": int(os.environ["PRED_LEN"]),
    "test_mse": mse,
    "test_mae": mae,
    "model": os.environ.get("MODEL_NAME", "TriModalLearnableWaveletPacketGatedProQwen"),
    "timestamp": now,
    "seed": int(os.environ["SEED"]),
    "seq_len": int(os.environ["SEQ_LEN"]),
    "channel": int(os.environ["CHANNEL"]),
    "batch_size": int(os.environ["BATCH_SIZE_RUN"]),
    "learning_rate": float(os.environ["LR"]),
    "dropout_n": float(os.environ["DROP"]),
    "weight_decay": float(os.environ["WEIGHT_DECAY"]),
    "wp_level": int(os.environ.get("WP_LEVEL", "2")),
}
print(json.dumps(data, ensure_ascii=False))
PY
}

run_one() {
  local pred_len="$1" lr="$2" channel="$3" dropout="$4" weight_decay="$5" batch_size="$6" epochs="$7"

  export DATA_PATH SEQ_LEN MODEL_NAME MODEL_ID WP_LEVEL LRADJ PATIENCE GPU
  export PRED_LEN="${pred_len}"
  export LR="${lr}"
  export CHANNEL="${channel}"
  export DROP="${dropout}"
  export WEIGHT_DECAY="${weight_decay}"
  export BATCH_SIZE_RUN="${batch_size}"

  for seed in $(seq 2020 2040); do
    export SEED="${seed}"

    log_file="${LOG_DIR}/i${SEQ_LEN}_o${pred_len}_lr${lr}_c${channel}_dn${dropout}_wd${weight_decay}_bs${batch_size}_seed${seed}.log"
    echo ">>>> 开始训练: pred_len=${pred_len}, seed=${seed}, lr=${lr}, c=${channel}, drop=${dropout}, wd=${weight_decay}, log=${log_file}"

    cmd=(python /root/0/T3Time/train_learnable_wavelet_packet_pro.py
      --model_id "${MODEL_ID}"
      --model "${MODEL_NAME}"
      --data_path "${DATA_PATH}"
      --pred_len "${pred_len}"
      --channel "${channel}"
      --batch_size "${batch_size}"
      --learning_rate "${lr}"
      --dropout_n "${dropout}"
      --weight_decay "${weight_decay}"
      --wp_level "${WP_LEVEL}"
      --lradj "${LRADJ}"
      --epochs "${epochs}"
      --patience "${PATIENCE}"
      --seed "${seed}"
      --gpu "${GPU}"
    )

    "${cmd[@]}" > "${log_file}" 2>&1

    # 如果训练脚本已经写入日志，这里作为备用解析
    # 实际上 train_learnable_wavelet_packet_pro.py 已经会自动写入 experiment_results.log
    # 但为了确保数据完整性，这里也解析一次
    if append_result "${log_file}" >> "${RESULT_LOG}" 2>/dev/null; then
      echo "已写入结果到 ${RESULT_LOG}"
    else
      echo "[WARN] 解析日志失败，但训练脚本可能已自动写入结果"
    fi
  done
}

for cfg in "${CONFIGS[@]}"; do
  run_one ${cfg}
done

echo "全部任务完成，结果已追加到 ${RESULT_LOG}"
