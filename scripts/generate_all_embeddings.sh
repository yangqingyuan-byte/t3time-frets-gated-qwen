#!/bin/bash
# 批量生成所有数据集的embeddings
# 使用版本标识避免覆盖不同版本的embeddings

export CUDA_VISIBLE_DEVICES=0

# 设置嵌入版本（可选: original, wavelet, gpt2等）
EMBED_VERSION=${1:-wavelet}  # 默认使用wavelet版本

data_paths=("ETTh1" "ETTh2" "ETTm1" "ETTm2")
divides=("train" "val" "test")

echo "=========================================="
echo "Batch generating embeddings"
echo "Embedding version: ${EMBED_VERSION}"
echo "=========================================="

for data_path in "${data_paths[@]}"; do
  for divide in "${divides[@]}"; do
    echo "=========================================="
    echo "Generating embeddings for ${data_path} ${divide}..."
    echo "Version: ${EMBED_VERSION}"
    echo "=========================================="
    python generate_embeddings_wavelet.py \
        --data_path $data_path \
        --divide $divide \
        --input_len 96 \
        --output_len 96 \
        --device cuda \
        --batch_size 1 \
        --num_workers 4 \
        --embed_version ${EMBED_VERSION}
    echo ""
  done
done

echo "All embeddings generated! Version: ${EMBED_VERSION}"

