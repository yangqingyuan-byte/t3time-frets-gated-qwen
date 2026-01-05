# 在 Screen 中运行参数寻优脚本

## 🎯 为什么使用 Screen？

- ✅ **持久运行**: 即使 SSH 连接断开，脚本也会继续运行
- ✅ **随时查看**: 可以随时连接查看进度
- ✅ **后台运行**: 不占用当前终端
- ✅ **多任务**: 可以同时运行多个 screen session

## 🚀 快速开始

### 方法1: 使用辅助脚本（推荐）

```bash
bash scripts/T3Time_FreTS_FusionExp/run_hyperopt_in_screen.sh
```

这个脚本会：
1. 检查并安装 screen（如果需要）
2. 检查是否有已存在的 session
3. 创建新的 screen session 并运行脚本
4. 提供连接选项

### 方法2: 手动创建 Screen Session

```bash
# 1. 创建并进入 screen session
screen -S frets_hyperopt

# 2. 在 screen 中运行脚本
cd /root/0/T3Time
bash scripts/T3Time_FreTS_FusionExp/ETTh1_hyperopt.sh

# 3. 分离 screen（不中断脚本运行）
# 按 Ctrl+A，然后按 D
```

## 📋 Screen 常用命令

### 基本操作

```bash
# 创建新的 screen session
screen -S session_name

# 列出所有 screen session
screen -ls

# 连接到指定的 session
screen -r session_name

# 如果 session 处于 Attached 状态，强制连接
screen -d -r session_name

# 杀死指定的 session
screen -S session_name -X quit
```

### Screen 内部快捷键

在 screen session 内部：

- **Ctrl+A, D**: 分离（detach）screen，脚本继续运行
- **Ctrl+A, K**: 杀死当前 window
- **Ctrl+A, C**: 创建新的 window
- **Ctrl+A, N**: 切换到下一个 window
- **Ctrl+A, P**: 切换到上一个 window
- **Ctrl+A, [**: 进入复制模式（可以滚动查看历史）
  - 在复制模式中：空格键开始选择，回车键复制，Esc 退出
- **Ctrl+A, ]**: 粘贴

## 🔍 监控脚本运行

### 查看 Screen Session 列表

```bash
screen -ls
```

输出示例：
```
There is a screen on:
        12345.frets_hyperopt    (Attached)
1 Socket in /var/run/screen/S-root.
```

### 连接到 Session 查看进度

```bash
screen -r frets_hyperopt
```

### 在 Screen 外部查看日志

```bash
# 查看最新的训练日志
ls -lt Results/T3Time_FreTS_FusionExp/ETTh1/*.log | head -5

# 查看结果日志的最新记录
tail -20 experiment_results.log

# 使用状态检查脚本
bash scripts/T3Time_FreTS_FusionExp/check_hyperopt_status.sh
```

## 🛠️ 故障排除

### 问题1: Screen Session 显示 "Attached" 但无法连接

**解决方案**:
```bash
# 强制分离并连接
screen -d -r frets_hyperopt
```

### 问题2: 忘记 Session 名称

**解决方案**:
```bash
# 列出所有 session
screen -ls

# 如果有多个，可以连接到任意一个
screen -r
```

### 问题3: 脚本在 Screen 中停止

**可能原因**:
- 单个实验失败（已修复，脚本会继续运行）
- CUDA 内存不足
- 磁盘空间不足

**检查方法**:
```bash
# 连接到 screen 查看
screen -r frets_hyperopt

# 或查看日志
tail -50 Results/T3Time_FreTS_FusionExp/ETTh1/*.log | grep -E "Error|Exception|Killed"
```

### 问题4: 需要重启脚本

**解决方案**:
```bash
# 1. 连接到 screen
screen -r frets_hyperopt

# 2. 停止当前脚本（Ctrl+C）

# 3. 重新运行
bash scripts/T3Time_FreTS_FusionExp/ETTh1_hyperopt.sh

# 4. 分离 screen（Ctrl+A, D）
```

## 📊 推荐工作流程

### 1. 启动脚本

```bash
# 使用辅助脚本
bash scripts/T3Time_FreTS_FusionExp/run_hyperopt_in_screen.sh

# 或手动创建
screen -S frets_hyperopt
cd /root/0/T3Time
bash scripts/T3Time_FreTS_FusionExp/ETTh1_hyperopt.sh
# 按 Ctrl+A, D 分离
```

### 2. 定期检查进度

```bash
# 方法1: 连接到 screen 查看
screen -r frets_hyperopt

# 方法2: 使用状态检查脚本
bash scripts/T3Time_FreTS_FusionExp/check_hyperopt_status.sh

# 方法3: 查看结果日志
tail -20 experiment_results.log
```

### 3. 查看结果

```bash
# 分析结果
python scripts/T3Time_FreTS_FusionExp/analyze_ablation_results.py

# 或使用筛选脚本
python 筛选分析实验结果.py
```

## 💡 提示

1. **命名规范**: 使用有意义的 session 名称，如 `frets_hyperopt`, `frets_ablation` 等
2. **定期检查**: 建议每天检查一次运行状态
3. **日志备份**: 定期备份重要的日志文件
4. **资源监控**: 使用 `nvidia-smi` 和 `htop` 监控资源使用

## 🔄 替代方案: tmux

如果你更喜欢使用 tmux：

```bash
# 安装 tmux
sudo apt-get install tmux  # 或 yum install tmux

# 创建 session
tmux new -s frets_hyperopt

# 运行脚本
cd /root/0/T3Time
bash scripts/T3Time_FreTS_FusionExp/ETTh1_hyperopt.sh

# 分离: Ctrl+B, 然后按 D
# 连接: tmux attach -t frets_hyperopt
```
