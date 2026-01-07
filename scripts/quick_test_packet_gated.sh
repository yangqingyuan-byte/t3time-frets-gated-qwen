#!/bin/bash

# 快速测试脚本：验证 T3Time_Wavelet_Packet_Gated_Qwen 模型及其日志记录
# 仅运行 1 个 epoch

echo "开始快速测试 T3Time_Wavelet_Packet_Gated_Qwen..."

python train_wavelet_packet_gated_qwen.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --num_nodes 7 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --dropout_n 0.3 \
  --channel 64 \
  --epochs 1 \
  --es_patience 1 \
  --embed_version qwen3_0.6b \
  --wavelet db4 \
  --wp_level 2 \
  --seed 2026

if [ $? -eq 0 ]; then
    echo -e "\n✅ 测试成功！模型运行和日志记录正常。"
    echo "您可以运行以下命令查看测试日志："
    echo "tail -n 1 experiment_results.log"
else
    echo -e "\n❌ 测试失败！请检查上述报错信息。"
fi

