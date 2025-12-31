#!/bin/bash
# 自动配置脚本：在进入 T3Time 目录时自动激活 conda 环境
# 运行方法: source setup_auto_env.sh

CONFIG_FILE="$HOME/.bashrc"
PROJECT_DIR="/root/0/T3Time"
ENV_NAME="TimeCMA_Qwen3"

# 检查是否已经配置过
if grep -q "T3Time auto activate conda" "$CONFIG_FILE" 2>/dev/null; then
    echo "⚠ 检测到已存在配置，跳过添加"
    echo "如需重新配置，请手动编辑 $CONFIG_FILE"
else
    # 添加自动激活函数到 .bashrc
    cat >> "$CONFIG_FILE" << 'EOF'

# ===== T3Time auto activate conda environment =====
# 自动激活 T3Time 项目的 conda 环境
auto_activate_t3time_env() {
    # 检查是否在 T3Time 项目目录中
    if [[ "$PWD" == /root/0/T3Time* ]] || [[ "$PWD" == "$HOME/0/T3Time"* ]]; then
        # 如果环境未激活或激活的不是目标环境
        if [[ "$CONDA_DEFAULT_ENV" != "TimeCMA_Qwen3" ]]; then
            # 初始化 conda（如果还没有）
            if [ -z "$CONDA_SHLVL" ] || [ "$CONDA_SHLVL" -eq 0 ]; then
                if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
                    source "$HOME/anaconda3/etc/profile.d/conda.sh" > /dev/null 2>&1
                elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
                    source "$HOME/miniconda3/etc/profile.d/conda.sh" > /dev/null 2>&1
                fi
            fi
            
            # 激活环境
            if command -v conda &> /dev/null; then
                conda activate TimeCMA_Qwen3 > /dev/null 2>&1
                if [ $? -eq 0 ]; then
                    echo "✓ 已自动激活 conda 环境: TimeCMA_Qwen3"
                fi
            fi
        fi
    # 如果不在项目目录中，且当前环境是 TimeCMA_Qwen3，可以选择自动退出（可选）
    # elif [[ "$CONDA_DEFAULT_ENV" == "TimeCMA_Qwen3" ]] && [[ "$PWD" != /root/0/T3Time* ]]; then
    #     conda deactivate > /dev/null 2>&1
    fi
}

# 将函数添加到 PROMPT_COMMAND（每次显示提示符前执行）
if [[ -z "$PROMPT_COMMAND" ]]; then
    export PROMPT_COMMAND="auto_activate_t3time_env"
else
    # 如果 PROMPT_COMMAND 已存在，追加而不是替换
    export PROMPT_COMMAND="auto_activate_t3time_env; $PROMPT_COMMAND"
fi
# ===== End T3Time auto activate =====
EOF

    echo "✓ 配置已添加到 $CONFIG_FILE"
    echo ""
    echo "请运行以下命令使配置生效："
    echo "  source $CONFIG_FILE"
    echo ""
    echo "或者重新打开终端，进入 /root/0/T3Time 目录时会自动激活环境"
fi

