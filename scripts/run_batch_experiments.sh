#!/bin/bash
# 批量运行实验脚本
# 可以同时运行多个不同配置的实验

# 设置实验参数数组
data_paths=("ETTh1")  # 可以添加多个数据集: ("ETTh1" "ETTh2" "ETTm1" "ETTm2")
pred_lens=(96 192 336 720)  # 不同的预测长度
channels=(64 128 256)  # 不同的通道数
experiment_tags=("wavelet_db4" "gpt2")  # 不同的实验标签

# 基础参数
seq_len=96
learning_rate=1e-4
e_layer=1
d_layer=1
dropout_n=0.4
batch_size=16
seed=2024

# 创建结果目录
log_path="./Results/"
mkdir -p "$log_path"

# 计数器
count=0

echo "=========================================="
echo "批量实验开始"
echo "=========================================="

for data_path in "${data_paths[@]}"; do
    for pred_len in "${pred_lens[@]}"; do
        for channel in "${channels[@]}"; do
            for experiment_tag in "${experiment_tags[@]}"; do
                count=$((count + 1))
                
                # 根据实验标签选择训练脚本
                if [[ "$experiment_tag" == *"wavelet"* ]]; then
                    train_script="train_wavelet.py"
                    wavelet="db4"
                    extra_args="--wavelet $wavelet --use_cross_attention"
                else
                    train_script="train.py"
                    extra_args=""
                fi
                
                # 生成日志文件名
                log_file="${log_path}${data_path}/i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}_seed${seed}_${experiment_tag}.log"
                
                # 创建数据集目录
                mkdir -p "${log_path}${data_path}"
                
                echo ""
                echo "[$count] 启动实验: ${data_path} - ${experiment_tag} - i${seq_len}_o${pred_len}_c${channel}"
                echo "日志文件: ${log_file}"
                
                # 运行训练
                nohup python $train_script \
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
                    $extra_args \
                    --save ./logs/ > "$log_file" 2>&1 &
                
                # 等待一段时间再启动下一个实验（避免GPU内存冲突）
                sleep 5
            done
        done
    done
done

echo ""
echo "=========================================="
echo "已启动 $count 个实验"
echo "使用以下命令查看运行状态:"
echo "  ps aux | grep train"
echo "  tail -f ${log_path}*/i*_o*.log"
echo "=========================================="

