# 参数寻优脚本使用说明

## 概述

在 seed=2088 上进行参数寻优，搜索最佳的超参数组合。

## 参数搜索空间

- **Channel**: [32, 64, 128, 256] (4个值)
- **Dropout**: [0.1, 0.2, 0.3, 0.4, 0.5] (5个值)
- **Head**: [2, 4, 6, 8, 10, 12, 14, 16] (8个值)

**总实验数**: 4 × 5 × 8 = **160 个实验**

## 固定参数

所有实验使用以下固定参数：
- `seed=2088`
- `seq_len=96`
- `pred_len=96`
- `batch_size=16`
- `learning_rate=0.0001`
- `weight_decay=1e-4`
- `e_layer=1`
- `d_layer=1`
- `epochs=100`
- `patience=10`
- `loss_fn=smooth_l1`
- `lradj=type1`
- `embed_version=qwen3_0.6b`

## 使用方法

### 1. 运行参数寻优

```bash
bash scripts/T3Time_FreTS_FusionExp/hyperopt_seed2088.sh
```

脚本会：
- 遍历所有参数组合（160个实验）
- 每个实验的结果自动追加到 `experiment_results.log`
- 日志文件保存在 `Results/T3Time_FreTS_Gated_Qwen_Hyperopt/ETTh1/`
- 允许单个实验失败时继续运行（不会中断整个流程）

### 2. 检索最佳参数组合

```bash
python scripts/T3Time_FreTS_FusionExp/find_best_params_seed2088.py
```

或指定结果文件：

```bash
python scripts/T3Time_FreTS_FusionExp/find_best_params_seed2088.py --result_file /path/to/results.log
```

## 检索脚本输出

检索脚本会输出：

1. **最佳参数组合（单次最佳）**
   - 显示 MSE 最低的单次实验结果
   - 包含 Channel, Dropout, Head, MSE, MAE 等信息

2. **Top 10 最佳配置**
   - 按 MSE 排序的前10个结果

3. **参数统计分析**
   - 每个参数组合的平均 MSE、最小 MSE、最大 MSE
   - 按平均 MSE 排序

4. **各参数维度分析**
   - Channel 参数分析：不同 Channel 值的平均性能
   - Dropout 参数分析：不同 Dropout 值的平均性能
   - Head 参数分析：不同 Head 值的平均性能

5. **最佳参数组合（按平均 MSE）**
   - 如果同一参数组合运行了多次，显示平均 MSE 最低的组合

## 运行时间估算

假设每个实验需要约 10-15 分钟：
- **总时间**: 160 × 12.5 分钟 ≈ **33 小时**

建议在 `screen` 会话中运行：

```bash
# 创建 screen 会话
screen -S hyperopt_2088

# 运行参数寻优
bash scripts/T3Time_FreTS_FusionExp/hyperopt_seed2088.sh

# 按 Ctrl+A 然后 D 来 detach
# 重新连接: screen -r hyperopt_2088
```

## 输出文件

- **实验结果**: `/root/0/T3Time/experiment_results.log`
- **训练日志**: `/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_Hyperopt/ETTh1/`

## 示例输出

检索脚本的示例输出：

```
================================================================================
T3Time_FreTS_Gated_Qwen 参数寻优结果分析 (Seed=2088)
================================================================================

找到 160 条实验结果

================================================================================
🏆 最佳参数组合（单次最佳）
================================================================================
Channel:     64
Dropout:     0.1
Head:        8
MSE:         0.370849
MAE:         0.391841
Seed:        2088
Timestamp:   2026-01-03 12:00:00

================================================================================
Top 10 最佳配置
================================================================================
Rank   Channel    Dropout    Head     MSE            MAE           
--------------------------------------------------------------------------------
1      64         0.1        8        0.370849       0.391841
2      64         0.1        6        0.371234       0.392156
...
```

## 注意事项

1. **实验数量**: 总共 160 个实验，需要较长时间完成
2. **GPU 资源**: 确保有足够的 GPU 资源（脚本使用 `CUDA_VISIBLE_DEVICES=1`）
3. **磁盘空间**: 确保有足够的磁盘空间存储日志文件
4. **中断恢复**: 如果脚本中断，可以重新运行，已完成的实验不会重复运行（但会重新训练）
5. **结果记录**: 所有结果会自动追加到 `experiment_results.log`，不会覆盖之前的结果

## 故障排除

### 未找到实验结果

如果检索脚本提示"未找到实验结果"，请检查：
1. 参数寻优脚本是否已完成
2. `experiment_results.log` 中是否有对应的结果
3. 模型ID前缀是否正确（默认: `T3Time_FreTS_Gated_Qwen_Hyperopt`）

### 实验失败

如果某些实验失败：
- 脚本会继续运行其他实验（使用了 `|| true`）
- 失败的实验不会记录到结果文件中
- 可以查看对应的日志文件了解失败原因
