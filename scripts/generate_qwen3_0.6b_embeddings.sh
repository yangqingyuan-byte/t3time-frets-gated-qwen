#!/bin/bash
# 生成 qwen3_0.6b 嵌入文件
# 用法: bash scripts/generate_qwen3_0.6b_embeddings.sh [data_path] [gpu_id]
# 示例: bash scripts/generate_qwen3_0.6b_embeddings.sh ETTh1 0

set -e

# 参数设置
DATA_PATH=${1:-ETTh1}  # 默认 ETTh1
GPU_ID=${2:-0}  # 默认 GPU 0

export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 激活 conda 环境（如果需要）
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 设置 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH-}"

# 切换到项目根目录
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "生成 Qwen3-0.6B 嵌入文件"
echo "数据集: ${DATA_PATH}"
echo "GPU: ${GPU_ID}"
echo "嵌入版本: qwen3_0.6b"
echo "=========================================="

# 生成训练集嵌入
echo ""
echo "1. 生成 TRAIN 嵌入..."
python storage/store_emb_qwen3_0.6b.py \
    --data_path ${DATA_PATH} \
    --divide train \
    --input_len 96 \
    --output_len 96 \
    --device cuda \
    --batch_size 1 \
    --num_workers 4 \
    --d_model 1024 \
    --l_layers 28 \
    --model_name "Qwen/Qwen3-0.6B" \
    --embed_version qwen3_0.6b

# 生成验证集嵌入
echo ""
echo "2. 生成 VAL 嵌入..."
python storage/store_emb_qwen3_0.6b.py \
    --data_path ${DATA_PATH} \
    --divide val \
    --input_len 96 \
    --output_len 96 \
    --device cuda \
    --batch_size 1 \
    --num_workers 4 \
    --d_model 1024 \
    --l_layers 28 \
    --model_name "Qwen/Qwen3-0.6B" \
    --embed_version qwen3_0.6b

# 生成测试集嵌入
echo ""
echo "3. 生成 TEST 嵌入..."
python storage/store_emb_qwen3_0.6b.py \
    --data_path ${DATA_PATH} \
    --divide test \
    --input_len 96 \
    --output_len 96 \
    --device cuda \
    --batch_size 1 \
    --num_workers 4 \
    --d_model 1024 \
    --l_layers 28 \
    --model_name "Qwen/Qwen3-0.6B" \
    --embed_version qwen3_0.6b

echo ""
echo "=========================================="
echo "✅ 所有嵌入文件生成完成！"
echo "保存路径: ./Embeddings/${DATA_PATH}/qwen3_0.6b/{train,val,test}/"
echo "=========================================="
