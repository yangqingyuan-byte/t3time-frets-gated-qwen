#!/bin/bash
# 将当前目录的所有文件复制到 /root/0/T3Time

SOURCE_DIR="/root/t3time-frets-gated-qwen"
TARGET_DIR="/root/0/T3Time"

echo "=========================================="
echo "复制文件: $SOURCE_DIR -> $TARGET_DIR"
echo "=========================================="
echo "源目录大小: $(du -sh $SOURCE_DIR | cut -f1)"
echo "开始复制..."
echo ""

# 方法1: 使用 rsync（推荐，支持进度显示和断点续传）
rsync -avh --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.specstory' \
    "$SOURCE_DIR/" "$TARGET_DIR/"

echo ""
echo "=========================================="
echo "复制完成！"
echo "=========================================="
echo "目标目录大小: $(du -sh $TARGET_DIR | cut -f1)"
