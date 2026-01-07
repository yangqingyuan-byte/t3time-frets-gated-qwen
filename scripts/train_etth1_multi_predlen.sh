#!/bin/bash
set -uo pipefail

unset __vsc_prompt_cmd_original 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

data_path="ETTh1"
seq_len=96
batch_size=16

log_path="./Results/${data_path}/"
mkdir -p $log_path

echo "=========================================="
echo "ETTh1 多预测长度训练"
echo "预测长度: 96, 192, 336, 720"
echo "=========================================="
echo ""

# pred_len 96
echo "[1/4] 训练: pred_len=96"
pred_len=96
learning_rate=1e-4
channel=256
e_layer=1
d_layer=1
dropout_n=0.4

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
python train.py \
  --data_path $data_path \
  --batch_size 256 \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 150 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer | tee $log_file

echo "✅ pred_len=96 训练完成"
echo ""

# pred_len 192
echo "[2/4] 训练: pred_len=192"
pred_len=192
learning_rate=1e-4
channel=256
e_layer=1
d_layer=2
dropout_n=0.6

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
python train.py \
  --data_path $data_path \
  --batch_size 32 \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 150 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer | tee $log_file

echo "✅ pred_len=192 训练完成"
echo ""

# pred_len 336
echo "[3/4] 训练: pred_len=336"
pred_len=336
learning_rate=1e-4
channel=64
dropout_n=0.7
e_layer=1
d_layer=2

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 120 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer | tee $log_file

echo "✅ pred_len=336 训练完成"
echo ""

# pred_len 720
echo "[4/4] 训练: pred_len=720"
pred_len=720
learning_rate=1e-4
channel=64
dropout_n=0.5
e_layer=3
d_layer=4

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
python train.py \
  --data_path $data_path \
  --batch_size 32 \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 150 \
  --seed 2024 \
  --channel $channel \
  --head 8 \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer | tee $log_file

echo "✅ pred_len=720 训练完成"
echo ""

echo "=========================================="
echo "✅ 所有训练完成！"
echo "=========================================="
echo "查看结果: python scripts/analyze_etth1_results.py"
echo "=========================================="
