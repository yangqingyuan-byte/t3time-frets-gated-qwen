#!/bin/bash
# 运行原始T3Time模型实验的脚本
# 自动生成符合规范的日志文件名

# 设置实验参数
data_path="ETTh1"
seq_len=96
pred_len=96
learning_rate=1e-4
channel=256
e_layer=1
d_layer=1
dropout_n=0.4
batch_size=16
seed=2024
experiment_tag="gpt2"  # 实验标签，用于区分不同实验

# 创建结果目录
log_path="./Results/${data_path}/"
mkdir -p "$log_path"

# 生成日志文件名（符合规范）
# 格式: i{seq_len}_o{pred_len}_lr{learning_rate}_c{channel}_el{e_layer}_dl{d_layer}_dn{dropout_n}_bs{batch_size}_seed{seed}_{experiment_tag}.log
log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}_seed${seed}_${experiment_tag}.log"

echo "=========================================="
echo "开始实验: ${experiment_tag}"
echo "日志文件: ${log_file}"
echo "=========================================="

# 运行训练并将输出重定向到日志文件
nohup python train.py \
    --data_path $data_path \
    --device cuda \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --channel $channel \
    --num_nodes 7 \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --dropout_n $dropout_n \
    --e_layer $e_layer \
    --d_layer $d_layer \
    --head 8 \
    --epochs 150 \
    --seed $seed \
    --embed_version original > "$log_file" 2>&1 &

# 获取进程ID
pid=$!
echo "实验已在后台运行，进程ID: $pid"
echo "查看日志: tail -f ${log_file}"
echo "停止实验: kill $pid"

