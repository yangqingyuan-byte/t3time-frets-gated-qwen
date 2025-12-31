#!/bin/bash
# 生成ETTh1数据集的embeddings（训练、验证、测试）
# 使用版本标识避免覆盖不同版本的embeddings

export CUDA_VISIBLE_DEVICES=0

# 设置嵌入版本（可选: original, wavelet, gpt2等）
EMBED_VERSION=${1:-wavelet}  # 默认使用wavelet版本

echo "=========================================="
echo "Generating embeddings for ETTh1 dataset"
echo "Embedding version: ${EMBED_VERSION}"
echo "=========================================="

# 生成训练集embeddings
echo ""
echo "1. Generating TRAIN embeddings..."
python generate_embeddings_wavelet.py \
    --data_path ETTh1 \
    --divide train \
    --input_len 96 \
    --output_len 96 \
    --device cuda \
    --batch_size 1 \
    --num_workers 4 \
    --embed_version ${EMBED_VERSION}

# 生成验证集embeddings
echo ""
echo "2. Generating VAL embeddings..."
python generate_embeddings_wavelet.py \
    --data_path ETTh1 \
    --divide val \
    --input_len 96 \
    --output_len 96 \
    --device cuda \
    --batch_size 1 \
    --num_workers 4 \
    --embed_version ${EMBED_VERSION}

# 生成测试集embeddings
echo ""
echo "3. Generating TEST embeddings..."
python generate_embeddings_wavelet.py \
    --data_path ETTh1 \
    --divide test \
    --input_len 96 \
    --output_len 96 \
    --device cuda \
    --batch_size 1 \
    --num_workers 4 \
    --embed_version ${EMBED_VERSION}

echo ""
echo "=========================================="
echo "All embeddings generated successfully!"
echo "=========================================="

