# T3Time_FreTS_FusionExp 参数寻优

## 📋 脚本说明

`ETTh1_hyperopt.sh` 是针对 `T3Time_FreTS_FusionExp` 模型的参数寻优脚本，在原始最佳配置基础上微调 `scale` 和 `sparsity_threshold` 参数。

## 🎯 基础配置（固定）

基于原始最佳结果（MSE=0.377742）的配置：
- **channel**: 64
- **dropout**: 0.1
- **weight_decay**: 1e-4
- **loss_fn**: smooth_l1
- **fusion_mode**: gate
- **affine**: True（模型代码中）

## 🔬 寻优参数

### Scale 参数（FreTS Component 初始化）
- **范围**: 0.010 - 0.025
- **重点**: 0.015, 0.018, 0.020（原始最佳）, 0.022, 0.025
- **理论**: 控制权重矩阵的初始化范围

### Sparsity Threshold 参数（频域稀疏化）
- **范围**: 0.008 - 0.015
- **重点**: 0.008, 0.009, 0.010（原始最佳）, 0.012, 0.015
- **理论**: 控制频域特征的稀疏化程度

## 📊 测试配置列表

脚本会测试以下组合：

1. **Scale 微调**（保持 sparsity_threshold=0.01）:
   - 0.015, 0.018, 0.020, 0.022, 0.025

2. **Sparsity Threshold 微调**（保持 scale=0.02）:
   - 0.008, 0.009, 0.010, 0.012, 0.015

3. **组合优化**:
   - (0.018, 0.009), (0.018, 0.010), (0.018, 0.012)
   - (0.022, 0.009), (0.022, 0.010), (0.022, 0.012)

4. **更小的 Scale**（基于之前发现 scale 越小性能越好的趋势）:
   - 0.010, 0.012, 0.014

**总计**: 约 20 个配置组合，每个配置对 seed 2020-2040 运行（21 个种子）

## 🚀 使用方法

### 1. 直接运行（前台）

```bash
bash scripts/T3Time_FreTS_FusionExp/ETTh1_hyperopt.sh
```

### 2. 后台运行（推荐）

```bash
nohup bash scripts/T3Time_FreTS_FusionExp/ETTh1_hyperopt.sh > frets_hyperopt.log 2>&1 &
```

### 3. 查看进度

```bash
# 查看后台任务
tail -f frets_hyperopt.log

# 查看已完成的训练数量
ls Results/T3Time_FreTS_FusionExp/ETTh1/*.log | wc -l

# 查看最新的结果
tail -20 experiment_results.log
```

## 📈 结果分析

训练完成后，所有结果都记录在 `experiment_results.log` 中。

### 快速查看最佳结果

```bash
# 查看所有 HyperOpt 的结果
grep "T3Time_FreTS_FusionExp_HyperOpt" experiment_results.log | \
  python -c "
import sys, json
results = []
for line in sys.stdin:
    data = json.loads(line.strip())
    results.append((
        data.get('frets_scale', 0),
        data.get('sparsity_threshold', 0),
        data['test_mse'],
        data['test_mae'],
        data.get('seed', 'unknown')
    ))
results.sort(key=lambda x: x[2])  # 按 MSE 排序
print('Top 10 最佳配置 (按 MSE 排序):')
print(f\"{'Scale':<8} {'Sparsity':<10} {'MSE':<12} {'MAE':<12} {'Seed':<8}\")
print('-' * 50)
for scale, sparsity, mse, mae, seed in results[:10]:
    print(f'{scale:<8.3f} {sparsity:<10.3f} {mse:<12.6f} {mae:<12.6f} {seed:<8}')
"
```

### 使用筛选脚本

```bash
python 筛选分析实验结果.py
# 然后选择: T3Time_FreTS_FusionExp_HyperOpt
```

## 🎯 预期结果

基于当前最佳结果（MSE=0.377142），预期：
- 可能找到 MSE < 0.377 的配置
- 最佳 scale 可能在 0.015-0.020 之间
- 最佳 sparsity_threshold 可能在 0.009-0.012 之间

## ⚙️ 自定义配置

编辑 `ETTh1_hyperopt.sh` 中的 `CONFIGS` 数组来修改搜索空间：

```bash
CONFIGS=(
  # 格式: "scale sparsity_threshold"
  "0.015 0.01"
  "0.020 0.01"
  # 添加更多配置...
)
```

## 📝 注意事项

1. 每个配置会运行 21 个种子（2020-2040），总训练时间较长
2. 建议使用后台运行，避免终端断开
3. 结果会自动追加到 `experiment_results.log`
4. 日志文件保存在 `Results/T3Time_FreTS_FusionExp/ETTh1/`

## 🔧 故障排除

如果遇到问题：

1. **CUDA 内存不足**: 调整 `CUDA_VISIBLE_DEVICES` 或减少并行任务
2. **日志解析失败**: 检查训练是否正常完成
3. **结果未写入**: 检查 `experiment_results.log` 的写入权限
