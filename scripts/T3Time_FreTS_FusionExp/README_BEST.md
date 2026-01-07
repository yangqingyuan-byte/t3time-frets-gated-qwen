# FreTS FusionExp 最佳配置训练

## 📋 脚本说明

`train_best_config.sh` 使用参数寻优得到的最佳配置进行训练，验证该配置在不同种子下的稳定性。

## 🎯 最佳配置参数

基于参数寻优结果（Top 1）：
- **Scale**: 0.018
- **Sparsity Threshold**: 0.009
- **最佳结果**: MSE=0.376336, MAE=0.390907 (seed 2021)

### 完整配置

```bash
# 最佳参数（寻优得到）
frets_scale=0.018
sparsity_threshold=0.009

# 基础配置（原始最佳）
channel=64
dropout=0.1
weight_decay=1e-4
loss_fn=smooth_l1
fusion_mode=gate
learning_rate=1e-4
pred_len=96
```

## 🚀 使用方法

### 1. 默认运行（种子 2020-2040）

```bash
bash scripts/T3Time_FreTS_FusionExp/train_best_config.sh
```

### 2. 自定义种子列表

```bash
# 运行指定种子
bash scripts/T3Time_FreTS_FusionExp/train_best_config.sh "2024 2025 2026"

# 运行单个种子
bash scripts/T3Time_FreTS_FusionExp/train_best_config.sh "2021"
```

### 3. 后台运行（推荐）

```bash
nohup bash scripts/T3Time_FreTS_FusionExp/train_best_config.sh > best_config_train.log 2>&1 &
```

## 📊 结果查看

### 快速查看结果

```bash
# 查看所有最佳配置的结果
grep "T3Time_FreTS_FusionExp_Best" experiment_results.log | \
  python -c "
import sys, json
results = []
for line in sys.stdin:
    data = json.loads(line.strip())
    results.append((data['seed'], data['test_mse'], data['test_mae']))
results.sort(key=lambda x: x[1])
print('最佳结果 (按 MSE 排序):')
for seed, mse, mae in results[:10]:
    print(f'  Seed {seed}: MSE={mse:.6f}, MAE={mae:.6f}')
if results:
    avg_mse = sum(r[1] for r in results) / len(results)
    avg_mae = sum(r[2] for r in results) / len(results)
    print(f'\n平均结果: MSE={avg_mse:.6f}, MAE={avg_mae:.6f}')
    print(f'最佳结果: MSE={results[0][1]:.6f}, MAE={results[0][2]:.6f} (Seed {results[0][0]})')
    print(f'最差结果: MSE={results[-1][1]:.6f}, MAE={results[-1][2]:.6f} (Seed {results[-1][0]})')
"
```

### 使用筛选脚本

```bash
python 筛选分析实验结果.py
# 然后选择: T3Time_FreTS_FusionExp_Best
```

## 📈 预期结果

基于参数寻优结果：
- **单次最佳**: MSE ≈ 0.376336
- **平均性能**: 预期在 0.376-0.378 之间
- **稳定性**: 不同种子下结果应该相对稳定

## 🎯 与原始最佳对比

| 配置 | Scale | Sparsity | MSE | MAE |
|------|-------|-----------|-----|-----|
| 原始最佳 | 0.020 | 0.010 | 0.377142 | 0.393041 |
| 寻优最佳 | 0.018 | 0.009 | 0.376336 | 0.390907 |
| **改进** | - | - | **-0.000806** | **-0.002134** |

改进幅度：
- MSE 改进: **0.21%**
- MAE 改进: **0.54%**

## 📝 注意事项

1. 训练时间：每个种子约 1-2 小时（取决于硬件）
2. 日志文件保存在 `Results/T3Time_FreTS_FusionExp_Best/ETTh1/`
3. 结果自动追加到 `experiment_results.log`
4. 建议使用后台运行，避免终端断开

## 🔧 故障排除

如果遇到问题：

1. **CUDA 内存不足**: 检查 `CUDA_VISIBLE_DEVICES` 设置
2. **日志解析失败**: 检查训练是否正常完成
3. **结果未写入**: 检查 `experiment_results.log` 的写入权限
