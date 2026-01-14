#!/bin/bash
# 微信通知测试脚本

echo "=========================================="
echo "微信通知测试脚本"
echo "=========================================="
echo ""

# 检查是否配置了 虾推啥(xtuis)
if [ -n "${SENDKEY:-}" ]; then
    echo "✅ 检测到 虾推啥(xtuis) 配置 (SENDKEY)"
    echo "正在发送测试消息..."
    python /root/0/T3Time/notify_wechat.py \
        --method serverchan \
        --sendkey "${SENDKEY}" \
        --title "T3Time 通知测试" \
        --body "这是一条测试消息

如果你收到这条消息，说明 虾推啥(xtuis) 配置成功！

时间: $(date '+%Y-%m-%d %H:%M:%S')
服务器: $(hostname)" && echo "✅ 测试消息已发送，请检查微信"
elif [ -n "${QYWX_CORPID:-}" ] && [ -n "${QYWX_CORPSECRET:-}" ] && [ -n "${QYWX_AGENTID:-}" ]; then
    echo "✅ 检测到企业微信配置"
    echo "正在发送测试消息..."
    python /root/0/T3Time/notify_wechat.py \
        --method qywx \
        --corpid "${QYWX_CORPID}" \
        --corpsecret "${QYWX_CORPSECRET}" \
        --agentid "${QYWX_AGENTID}" \
        --title "T3Time 通知测试" \
        --body "这是一条测试消息

如果你收到这条消息，说明企业微信配置成功！

时间: $(date '+%Y-%m-%d %H:%M:%S')
服务器: $(hostname)" && echo "✅ 测试消息已发送，请检查微信"
else
    echo "❌ 未检测到任何通知配置"
    echo ""
    echo "请选择一种方式配置："
    echo ""
    echo "【方式一：虾推啥(xtuis)（推荐）】"
    echo "1. 打开 https://wx.xtuis.cn/ 或官方入口，登录并获取 token"
    echo "2. 复制你的 token"
    echo "3. 运行: export SENDKEY='你的token'"
    echo "4. 再次运行此脚本测试"
    echo ""
    echo "【方式二：企业微信】"
    echo "1. 注册企业微信：https://work.weixin.qq.com/"
    echo "2. 创建应用，获取 CorpID, CorpSecret, AgentID"
    echo "3. 运行以下命令："
    echo "   export QYWX_CORPID='你的企业ID'"
    echo "   export QYWX_CORPSECRET='你的应用Secret'"
    echo "   export QYWX_AGENTID='你的应用AgentID'"
    echo "4. 再次运行此脚本测试"
    echo ""
    echo "详细配置说明请查看: /root/0/T3Time/微信通知配置说明.md"
    exit 1
fi
