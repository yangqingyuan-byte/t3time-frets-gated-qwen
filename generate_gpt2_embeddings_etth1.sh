#!/bin/bash
# 生成 ETTh1 数据集的 GPT-2 嵌入（训练、验证、测试集）
# 使用方法: bash generate_gpt2_embeddings_etth1.sh

echo "=========================================="
echo "生成 ETTh1 数据集的 GPT-2 嵌入"
echo "=========================================="

# 设置参数
DATA_PATH="ETTh1"
EMBED_VERSION="original"
DEVICE="cuda"
INPUT_LEN=96
OUTPUT_LEN=96
BATCH_SIZE=1
D_MODEL=768

echo "数据集: $DATA_PATH"
echo "嵌入版本: $EMBED_VERSION"
echo "设备: $DEVICE"
echo "输入长度: $INPUT_LEN"
echo "输出长度: $OUTPUT_LEN"
echo "=========================================="

# 生成训练集嵌入
echo ""
echo "=========================================="
echo ">>> 开始生成训练集嵌入..."
echo "=========================================="
python storage/store_emb_gpt2_etth1.py \
    --data_path $DATA_PATH \
    --divide train \
    --embed_version $EMBED_VERSION \
    --device $DEVICE \
    --input_len $INPUT_LEN \
    --output_len $OUTPUT_LEN \
    --batch_size $BATCH_SIZE \
    --d_model $D_MODEL

if [ $? -ne 0 ]; then
    echo "错误: 训练集嵌入生成失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo ">>> 训练集嵌入生成完成！"
echo "=========================================="
echo ""

# 生成验证集嵌入
echo "=========================================="
echo ">>> 开始生成验证集嵌入..."
echo "=========================================="
python storage/store_emb_gpt2_etth1.py \
    --data_path $DATA_PATH \
    --divide val \
    --embed_version $EMBED_VERSION \
    --device $DEVICE \
    --input_len $INPUT_LEN \
    --output_len $OUTPUT_LEN \
    --batch_size $BATCH_SIZE \
    --d_model $D_MODEL

if [ $? -ne 0 ]; then
    echo "错误: 验证集嵌入生成失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo ">>> 验证集嵌入生成完成！"
echo "=========================================="
echo ""

# 生成测试集嵌入
echo "=========================================="
echo ">>> 开始生成测试集嵌入..."
echo "=========================================="
python storage/store_emb_gpt2_etth1.py \
    --data_path $DATA_PATH \
    --divide test \
    --embed_version $EMBED_VERSION \
    --device $DEVICE \
    --input_len $INPUT_LEN \
    --output_len $OUTPUT_LEN \
    --batch_size $BATCH_SIZE \
    --d_model $D_MODEL

if [ $? -ne 0 ]; then
    echo "错误: 测试集嵌入生成失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo ">>> 测试集嵌入生成完成！"
echo "=========================================="
echo ""

echo "=========================================="
echo "所有嵌入生成完成！"
echo "=========================================="
echo "嵌入文件保存在: ./Embeddings/$DATA_PATH/$EMBED_VERSION/"
echo "  - train/ : 训练集嵌入"
echo "  - val/   : 验证集嵌入"
echo "  - test/  : 测试集嵌入"
echo "=========================================="
