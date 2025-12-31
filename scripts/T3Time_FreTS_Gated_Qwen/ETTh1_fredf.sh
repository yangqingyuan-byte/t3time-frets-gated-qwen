#!/bin/bash
# 说明：
# 1) 针对四个 pred_len（96/192/336/720）做网格参数寻优，每个配置对 seed 2020-2070 逐一运行；
# 2) 使用 FreDFLoss 进行训练；
# 3) 训练结束后解析日志末尾的 “On average horizons, Test MSE/MAE”，以 JSONL 追加到 experiment_results.log；

set -euo pipefail

export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTHONPATH="/root/0/T3Time/plug_and_play:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1 # 使用 1 号显卡

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_FreDF/ETTh1"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

DATA_PATH="ETTh1"
SEQ_LEN=96
MODEL_NAME="T3Time_FreTS_Gated_Qwen_FreDF"
EMBED_VERSION="qwen3_0.6b"

# 配置列表 (pred_len lr channel e_layer d_layer dropout_n batch_size epochs head_override num_nodes lambda_freq)
CONFIGS=(
  # pred_len 96
  "96 1e-4 256 1 1 0.3 16 150 - 7 0.5"
  "96 5e-5 256 1 1 0.4 16 150 - 7 0.3"
  
  # pred_len 192
  "192 1e-4 256 1 2 0.6 32 150 - 7 0.5"
  "192 5e-5 256 2 2 0.5 32 150 - 7 0.3"

  # pred_len 336
  "336 1e-4 128 1 2 0.7 16 120 - 7 0.5"
  "336 7e-5 128 2 2 0.6 16 120 - 7 0.3"

  # pred_len 720
  "720 1e-4 128 3 4 0.5 32 150 8 7 0.5"
  "720 7e-5 128 3 3 0.6 32 150 8 7 0.3"
)

append_result() {
  local log_file="$1"
  python - <<'PY' "${log_file}"
import json, os, re, sys, datetime
log_file = sys.argv[1]
try:
    text = open(log_file, 'r', encoding='utf-8', errors='ignore').read().splitlines()
except Exception as e:
    sys.exit(f"Read error: {e}")

mse = mae = None
for line in reversed(text):
    if "On average horizons" in line:
        m = re.search(r"Test MSE:\s*([0-9.]+),\s*Test MAE:\s*([0-9.]+)", line)
        if m:
            mse, mae = float(m.group(1)), float(m.group(2))
            break

if mse is None or mae is None:
    raise SystemExit(f"[WARN] 未在日志中找到测试指标: {log_file}")

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data = {
    "data_path": os.environ["DATA_PATH"],
    "pred_len": int(os.environ["PRED_LEN"]),
    "test_mse": mse,
    "test_mae": mae,
    "model": os.environ["MODEL_NAME"],
    "timestamp": now,
    "seed": int(os.environ["SEED"]),
    "seq_len": int(os.environ["SEQ_LEN"]),
    "channel": int(os.environ["CHANNEL"]),
    "batch_size": int(os.environ["BATCH_SIZE_RUN"]),
    "learning_rate": float(os.environ["LR"]),
    "dropout_n": float(os.environ["DROP"]),
    "e_layer": int(os.environ["E_LAYER"]),
    "d_layer": int(os.environ["D_LAYER"]),
    "lambda_freq": float(os.environ["LAMBDA_FREQ"]),
}
print(json.dumps(data, ensure_ascii=False))
PY
}

run_one() {
  local pred_len="$1" lr="$2" channel="$3" e_layer="$4" d_layer="$5" dropout="$6" batch_size="$7" epochs="$8" head_override="$9" num_nodes="${10}" lambda_freq="${11}"

  export DATA_PATH SEQ_LEN MODEL_NAME
  export PRED_LEN="${pred_len}"
  export LR="${lr}"
  export CHANNEL="${channel}"
  export DROP="${dropout}"
  export E_LAYER="${e_layer}"
  export D_LAYER="${d_layer}"
  export BATCH_SIZE_RUN="${batch_size}"
  export LAMBDA_FREQ="${lambda_freq}"

  for seed in $(seq 2020 2070); do
    export SEED="${seed}"

    log_file="${LOG_DIR}/i${SEQ_LEN}_o${pred_len}_lr${lr}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout}_bs${batch_size}_lf${lambda_freq}_seed${seed}.log"
    echo ">>>> 开始训练: pred_len=${pred_len}, seed=${seed}, lr=${lr}, c=${channel}, lf=${lambda_freq}, log=${log_file}"

    cmd=(python /root/0/T3Time/train_frets_gated_qwen_fredf.py
      --data_path "${DATA_PATH}"
      --batch_size "${batch_size}"
      --num_nodes "${num_nodes}"
      --seq_len "${SEQ_LEN}"
      --pred_len "${pred_len}"
      --epochs "${epochs}"
      --seed "${seed}"
      --channel "${channel}"
      --learning_rate "${lr}"
      --dropout_n "${dropout}"
      --e_layer "${e_layer}"
      --d_layer "${d_layer}"
      --embed_version "${EMBED_VERSION}"
      --lambda_freq "${lambda_freq}"
    )
    if [[ "${head_override}" != "-" ]]; then
      cmd+=(--head "${head_override}")
    fi

    "${cmd[@]}" > "${log_file}" 2>&1

    append_result "${log_file}" >> "${RESULT_LOG}"
    echo "已写入结果到 ${RESULT_LOG}"
  done
}

for cfg in "${CONFIGS[@]}"; do
  run_one ${cfg}
done

echo "全部任务完成，结果已追加到 ${RESULT_LOG}"

