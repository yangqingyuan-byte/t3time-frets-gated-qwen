#!/bin/bash
# 说明：逐个 seed（2020-2040）跑四个 pred_len 配置，训练完成后解析日志末尾的
# “On average horizons, Test MSE/MAE”并以 JSONL 追加到 /root/0/T3Time/experiment_results.log
# 如需后台并行可自行加上 nohup &，当前脚本为顺序运行，便于稳定记录结果。

set -euo pipefail

# 若未设置 PYTHONPATH，使用空串占位以避免 set -u 报错
#export PYTHONPATH="/mnt/d/Monaf/Personal/Time_series_forecasting/T3Time:${PYTHONPATH-}"
export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="/root/0/T3Time/Results/ETTh1"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

DATA_PATH="ETTh1"
SEQ_LEN=96

# 配置列表：pred_len learning_rate channel e_layer d_layer dropout_n batch_size epochs head_override(optional)
CONFIGS=(
  "96 1e-4 256 1 1 0.4 16 150 -"
  "192 1e-4 256 1 2 0.6 32 150 -"
  "336 1e-4 64  1 2 0.7 16 120 -"
  "720 1e-4 64  3 4 0.5 32 150 8"
)

append_result() {
  local log_file="$1"
  python - <<'PY' "${log_file}"
import json, os, re, sys, datetime
log_file = sys.argv[1]
text = open(log_file, 'r', encoding='utf-8', errors='ignore').read().splitlines()

mse = mae = None
for line in reversed(text):
    if "On average horizons" in line:
        m = re.search(r"Test MSE:\s*([0-9.]+),\s*Test MAE:\s*([0-9.]+)", line)
        if m:
            mse, mae = float(m.group(1)), float(m.group(2))
            break

model = None
for line in text[:5]:
    m = re.search(r"model_name='([^']+)'", line)
    if m:
        model = m.group(1)
        break

if mse is None or mae is None:
    raise SystemExit(f"[WARN] 未在日志中找到测试指标: {log_file}")

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data = {
    "data_path": os.environ["DATA_PATH"],
    "pred_len": int(os.environ["PRED_LEN"]),
    "test_mse": mse,
    "test_mae": mae,
    "model": model or "unknown",
    "timestamp": now,
    "seed": int(os.environ["SEED"]),
    "seq_len": int(os.environ["SEQ_LEN"]),
    "channel": int(os.environ["CHANNEL"]),
    "batch_size": int(os.environ["BATCH_SIZE_RUN"]),
    "learning_rate": float(os.environ["LR"]),
    "dropout_n": float(os.environ["DROP"]),
    "e_layer": int(os.environ["E_LAYER"]),
    "d_layer": int(os.environ["D_LAYER"]),
}
print(json.dumps(data, ensure_ascii=False))
PY
}

run_one() {
  local pred_len="$1" lr="$2" channel="$3" e_layer="$4" d_layer="$5" dropout="$6" batch_size="$7" epochs="$8" head_override="$9"

  export DATA_PATH SEQ_LEN
  export PRED_LEN="${pred_len}"
  export LR="${lr}"
  export CHANNEL="${channel}"
  export DROP="${dropout}"
  export E_LAYER="${e_layer}"
  export D_LAYER="${d_layer}"
  export BATCH_SIZE_RUN="${batch_size}"

  for seed in $(seq 2020 2040); do
    export SEED="${seed}"

    log_file="${LOG_DIR}/i${SEQ_LEN}_o${pred_len}_lr${lr}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout}_bs${batch_size}_seed${seed}.log"
    echo ">>>> 开始训练: pred_len=${pred_len}, seed=${seed}, log=${log_file}"

    cmd=(python /root/0/T3Time/train.py
      --data_path "${DATA_PATH}"
      --batch_size "${batch_size}"
      --num_nodes 7
      --seq_len "${SEQ_LEN}"
      --pred_len "${pred_len}"
      --epochs "${epochs}"
      --seed "${seed}"
      --channel "${channel}"
      --learning_rate "${lr}"
      --dropout_n "${dropout}"
      --e_layer "${e_layer}"
      --d_layer "${d_layer}"
    )
  if [[ "${head_override}" != "-" ]]; then
      cmd+=(--head "${head_override}")
    fi

    "${cmd[@]}" > "${log_file}"

    append_result "${log_file}" >> "${RESULT_LOG}"
    echo "已写入结果到 ${RESULT_LOG}"
  done
}

for cfg in "${CONFIGS[@]}"; do
  run_one ${cfg}
done

echo "全部任务完成，结果已追加到 ${RESULT_LOG}"