#!/bin/bash
# 激活 conda 环境的脚本
# 使用方法: source activate_env.sh

# 初始化 conda（如果还没有初始化）
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    # 尝试找到 conda
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    fi
fi

# 激活 conda 环境
if command -v conda &> /dev/null; then
    conda activate TimeCMA_Qwen3
    echo "✓ 已激活 conda 环境: TimeCMA_Qwen3"
    echo "当前环境: $CONDA_DEFAULT_ENV"
else
    echo "⚠ 错误: 未找到 conda 命令，请确保 conda 已正确安装"
    return 1
fi

