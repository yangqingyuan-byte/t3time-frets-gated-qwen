#!/bin/bash
# T3Time_FreTS_FusionExp 参数寻优脚本
# 说明：
# 1) 针对四个 pred_len（96/192/336/720）做网格参数寻优，每个配置对 seed 2020-2040 逐一运行；
# 2) 训练结束后解析日志末尾的 "On average horizons, Test MSE/MAE"，以 JSONL 追加到 /root/0/T3Time/experiment_results.log；
# 3) 默认顺序运行，确保记录稳定。若需并行可自行在外层加 nohup &。
# 4) 固定最佳参数：scale=0.018, sparsity_threshold=0.009, fusion_mode=gate

set -uo pipefail  # 移除 -e，允许单个实验失败时继续运行

# 若未设置 PYTHONPATH，用当前项目路径占位
export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_FusionExp/ETTh1"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

DATA_PATH="ETTh1"
SEQ_LEN=96
MODEL_NAME="T3Time_FreTS_Gated_Qwen_FusionExp"
MODEL_ID="T3Time_FreTS_FusionExp"
EMBED_VERSION="qwen3_0.6b"

# 固定最佳参数（基于参数寻优结果）
BEST_SCALE=0.018
BEST_SPARSITY=0.009
FUSION_MODE="gate"
LOSS_FN="smooth_l1"

# 配置列表：
# pred_len lr channel e_layer d_layer dropout_n weight_decay batch_size epochs
# 可按需增删组合以扩大/缩小搜索空间
CONFIGS=(
  # pred_len 96 - 基础配置
  "96 1e-4 64 1 1 0.1 1e-4 16 100"
  "96 1e-4 128 1 1 0.1 1e-4 16 100"
  "96 5e-5 64 1 1 0.1 1e-4 16 100"
  "96 1e-4 64 2 1 0.1 1e-4 16 100"
  "96 1e-4 64 1 2 0.1 1e-4 16 100"
  
  # pred_len 192
  "192 1e-4 64 1 1 0.1 1e-4 16 100"
  "192 1e-4 128 1 1 0.1 1e-4 16 100"
  "192 5e-5 64 1 2 0.1 1e-4 16 100"
  "192 1e-4 64 2 2 0.1 1e-4 16 100"
  
  # pred_len 336
  "336 1e-4 64 1 1 0.1 1e-4 16 100"
  "336 1e-4 128 1 2 0.1 1e-4 16 100"
  "336 5e-5 64 1 2 0.1 1e-4 16 100"
  
  # pred_len 720
  "720 1e-4 64 1 2 0.1 1e-4 16 100"
  "720 1e-4 128 2 2 0.1 1e-4 16 100"
  "720 5e-5 64 2 2 0.1 1e-4 16 100"
)

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

model_id = os.environ.get("MODEL_ID", "T3Time_FreTS_FusionExp")
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
    "e_layer": int(os.environ["E_LAYER"]),
    "d_layer": int(os.environ["D_LAYER"]),
    "fusion_mode": os.environ["FUSION_MODE"],
    "loss_fn": os.environ["LOSS_FN"],
    "frets_scale": float(os.environ["FRETS_SCALE"]),
    "sparsity_threshold": float(os.environ["SPARSITY_THRESHOLD"]),
    "lradj": os.environ.get("LRADJ", "type1"),
    "patience": int(os.environ.get("PATIENCE", "10")),
    "gpu": int(os.environ.get("GPU", "0")),
    "embed_version": os.environ["EMBED_VERSION"]
}
print(json.dumps(data, ensure_ascii=False))
PY
}

run_one() {
  local pred_len="$1" lr="$2" channel="$3" e_layer="$4" d_layer="$5" dropout="$6" weight_decay="$7" batch_size="$8" epochs="$9"

  export DATA_PATH SEQ_LEN MODEL_ID MODEL_NAME EMBED_VERSION
  export PRED_LEN="${pred_len}"
  export LR="${lr}"
  export CHANNEL="${channel}"
  export DROP="${dropout}"
  export WEIGHT_DECAY="${weight_decay}"
  export E_LAYER="${e_layer}"
  export D_LAYER="${d_layer}"
  export BATCH_SIZE_RUN="${batch_size}"
  export FUSION_MODE="${FUSION_MODE}"
  export LOSS_FN="${LOSS_FN}"
  export FRETS_SCALE="${BEST_SCALE}"
  export SPARSITY_THRESHOLD="${BEST_SPARSITY}"
  export LRADJ="type1"
  export PATIENCE=10
  export GPU=0

  # 统计信息
  total_seeds=$(seq 2020 2090 | wc -l)
  current_seed=0
  success_count=0
  fail_count=0

  for seed in $(seq 2020 2090); do
    current_seed=$((current_seed + 1))
    export SEED="${seed}"

    log_file="${LOG_DIR}/i${SEQ_LEN}_o${pred_len}_lr${lr}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout}_wd${weight_decay}_bs${batch_size}_seed${seed}.log"
    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始训练: pred_len=${pred_len}, seed=${seed}"
    echo "  配置: lr=${lr}, c=${channel}, el=${e_layer}, dl=${d_layer}, drop=${dropout}, wd=${weight_decay}"
    echo "  日志: ${log_file}"
    echo "=========================================="

    cmd=(python /root/0/T3Time/train_frets_gated_qwen_fusion_exp.py
      --data_path "${DATA_PATH}"
      --batch_size "${batch_size}"
      --seq_len "${SEQ_LEN}"
      --pred_len "${pred_len}"
      --epochs "${epochs}"
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
      --model_id "${MODEL_ID}"
    )

    # 运行训练，捕获退出码
    if "${cmd[@]}" > "${log_file}" 2>&1; then
      # 训练成功，尝试解析结果
      if append_result "${log_file}" >> "${RESULT_LOG}" 2>/dev/null; then
        success_count=$((success_count + 1))
        echo "✅ [${current_seed}/${total_seeds}] 完成: seed=${seed}, 结果已写入 ${RESULT_LOG}"
      else
        fail_count=$((fail_count + 1))
        echo "⚠️  [${current_seed}/${total_seeds}] 训练完成但解析失败: seed=${seed}, 查看日志: ${log_file}"
      fi
    else
      # 训练失败
      exit_code=$?
      fail_count=$((fail_count + 1))
      echo "❌ [${current_seed}/${total_seeds}] 训练失败: seed=${seed}, 退出码=${exit_code}, 查看日志: ${log_file}"
      # 继续运行下一个实验，不中断整个脚本
    fi
    
    # 每10个种子输出一次进度
    if [ $((current_seed % 10)) -eq 0 ]; then
      echo ""
      echo "进度: ${current_seed}/${total_seeds} | 成功: ${success_count} | 失败: ${fail_count}"
      echo ""
    fi
  done
  
  echo ""
  echo "配置完成统计: 成功=${success_count}, 失败=${fail_count}, 总计=${total_seeds}"
  echo ""
}

total_configs=${#CONFIGS[@]}
current_config=0

for cfg in "${CONFIGS[@]}"; do
  current_config=$((current_config + 1))
  echo ""
  echo "=========================================="
  echo "配置 ${current_config}/${total_configs}: ${cfg}"
  echo "=========================================="
  echo ""
  run_one ${cfg}
done

echo ""
echo "=========================================="
echo "✅ 全部任务完成！"
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
echo "    results.append((data['pred_len'], data['seed'], data['test_mse'], data['test_mae']))"
echo "results.sort(key=lambda x: (x[0], x[2]))"
echo "print('最佳结果 (按 pred_len 和 MSE 排序):')"
echo "for pred, seed, mse, mae in results[:10]:"
echo "    print(f'  Pred={pred}, Seed={seed}: MSE={mse:.6f}, MAE={mae:.6f}')"
echo "\""
