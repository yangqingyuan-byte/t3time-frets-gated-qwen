#!/bin/bash
# 在 screen 中运行参数寻优脚本的辅助脚本

SCRIPT_DIR="/root/0/T3Time/scripts/T3Time_FreTS_FusionExp"
SCRIPT_NAME="ETTh1_hyperopt.sh"
SCREEN_NAME="frets_hyperopt"

echo "=========================================="
echo "在 Screen 中运行参数寻优脚本"
echo "=========================================="
echo ""

# 检查 screen 是否安装
if ! command -v screen &> /dev/null; then
    echo "❌ screen 未安装，正在安装..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y screen
    elif command -v yum &> /dev/null; then
        sudo yum install -y screen
    else
        echo "请手动安装 screen: apt-get install screen 或 yum install screen"
        exit 1
    fi
fi

# 检查是否已有同名的 screen session
if screen -list | grep -q "${SCREEN_NAME}"; then
    echo "⚠️  发现已存在的 screen session: ${SCREEN_NAME}"
    echo "选项："
    echo "  1. 连接到现有 session"
    echo "  2. 杀死现有 session 并创建新的"
    echo "  3. 使用不同的名称创建新的 session"
    read -p "请选择 (1/2/3): " choice
    
    case $choice in
        1)
            echo "连接到现有 session..."
            screen -r "${SCREEN_NAME}"
            exit 0
            ;;
        2)
            echo "杀死现有 session..."
            screen -S "${SCREEN_NAME}" -X quit
            sleep 1
            ;;
        3)
            read -p "请输入新的 session 名称: " SCREEN_NAME
            ;;
        *)
            echo "取消操作"
            exit 0
            ;;
    esac
fi

# 创建新的 screen session 并运行脚本
echo "创建新的 screen session: ${SCREEN_NAME}"
echo "运行脚本: ${SCRIPT_DIR}/${SCRIPT_NAME}"
echo ""
echo "=========================================="
echo "Screen 使用说明："
echo "=========================================="
echo "  连接: screen -r ${SCREEN_NAME}"
echo "  分离: Ctrl+A, 然后按 D"
echo "  查看所有 session: screen -ls"
echo "  杀死 session: screen -S ${SCREEN_NAME} -X quit"
echo "=========================================="
echo ""

# 切换到脚本目录并运行
cd /root/0/T3Time
screen -dmS "${SCREEN_NAME}" bash -c "cd ${SCRIPT_DIR} && bash ${SCRIPT_NAME}; exec bash"

sleep 2

# 检查 session 是否创建成功
if screen -list | grep -q "${SCREEN_NAME}"; then
    echo "✅ Screen session 创建成功！"
    echo ""
    echo "立即连接到 session？(y/n)"
    read -p "> " connect_now
    
    if [ "${connect_now}" = "y" ] || [ "${connect_now}" = "Y" ]; then
        screen -r "${SCREEN_NAME}"
    else
        echo ""
        echo "稍后可以使用以下命令连接："
        echo "  screen -r ${SCREEN_NAME}"
    fi
else
    echo "❌ Screen session 创建失败"
    exit 1
fi
