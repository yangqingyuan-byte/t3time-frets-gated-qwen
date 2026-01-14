#!/bin/bash
# 修复 Git 作者信息，确保 commit 显示为你的 GitHub 账号

echo "=========================================="
echo "修复 Git 作者信息"
echo "=========================================="
echo ""

# 当前配置
echo "当前 Git 配置："
echo "  user.name: $(git config user.name)"
echo "  user.email: $(git config user.email)"
echo ""

# 选项1：使用 GitHub noreply 邮箱（推荐，无需验证）
GITHUB_NOREPLY="${1:-yangqingyuan-byte@users.noreply.github.com}"

echo "选项1: 使用 GitHub noreply 邮箱（推荐）"
echo "  邮箱: ${GITHUB_NOREPLY}"
echo "  优点: 无需验证，GitHub 自动识别"
echo ""

# 选项2：使用 QQ 邮箱（需要先在 GitHub 验证）
QQ_EMAIL="1124998618@qq.com"
echo "选项2: 使用 QQ 邮箱（需要先在 GitHub 验证）"
echo "  邮箱: ${QQ_EMAIL}"
echo "  步骤: 访问 https://github.com/settings/emails 添加并验证此邮箱"
echo ""

read -p "请选择 (1/2) [默认: 1]: " choice
choice=${choice:-1}

if [ "$choice" == "1" ]; then
    NEW_EMAIL="${GITHUB_NOREPLY}"
    echo ""
    echo "✅ 使用 GitHub noreply 邮箱: ${NEW_EMAIL}"
elif [ "$choice" == "2" ]; then
    NEW_EMAIL="${QQ_EMAIL}"
    echo ""
    echo "✅ 使用 QQ 邮箱: ${NEW_EMAIL}"
    echo "⚠️  请确保此邮箱已在 GitHub 验证！"
else
    echo "❌ 无效选择"
    exit 1
fi

# 修改 git 配置
echo ""
echo "正在修改 Git 配置..."
git config user.email "${NEW_EMAIL}"
echo "✅ Git 邮箱已更新为: ${NEW_EMAIL}"

# 显示新配置
echo ""
echo "新的 Git 配置："
echo "  user.name: $(git config user.name)"
echo "  user.email: $(git config user.email)"
echo ""

# 询问是否修改最近的 commit
read -p "是否修改最近的 commit 作者信息？(y/n) [默认: n]: " modify_commit
modify_commit=${modify_commit:-n}

if [ "$modify_commit" == "y" ] || [ "$modify_commit" == "Y" ]; then
    echo ""
    echo "正在修改最近的 commit..."
    git commit --amend --reset-author --no-edit
    echo "✅ 最近的 commit 作者信息已更新"
    echo ""
    echo "⚠️  注意: 如果已经 push 过，需要使用 'git push -f' 强制推送"
    echo "    (仅在你确定没有其他人基于此 commit 开发时使用)"
fi

echo ""
echo "=========================================="
echo "✅ 配置完成！"
echo "=========================================="
echo ""
echo "后续的 commit 将显示为你的 GitHub 账号"
echo "如果 GitHub 上仍然显示错误，请确保："
echo "1. 邮箱已在 GitHub 账号中验证"
echo "2. 访问 https://github.com/settings/emails 检查邮箱状态"
echo ""
