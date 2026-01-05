# T3Time_FreTS_Gated_Qwen_LongSeq 长序列优化模型

## 概述

这是针对长序列预测优化的改进版本模型，主要解决了原版 `T3Time_FreTS_Gated_Qwen` 在长序列（192, 336, 720）预测时性能较差的问题。

## 主要改进

### 1. 改进 Horizon 归一化
- **原版**: `pred_len / 100.0` (720 → 7.2，数值过大)
- **改进**: `pred_len / 1000.0` (720 → 0.72，更合理)
- **效果**: 避免 gate 饱和，保持梯度稳定

### 2. 改进融合公式
- **原版**: `time + gate * freq` (残差式，主要依赖时域)
- **改进**: `gate * freq + (1 - gate) * time` (加权平均)
- **效果**: 保证时域和频域都有贡献，更适合长序列

### 3. 动态稀疏化阈值
- **原版**: 固定 `sparsity_threshold=0.009`
- **改进**: 根据预测长度动态调整 `sparsity_threshold * (96 / pred_len)`
- **效果**: 长序列时降低阈值，保留更多频域信息

### 4. 全局上下文门控
- **原版**: 基于局部特征的门控
- **改进**: 使用 `RichHorizonGate`，基于全局池化
- **效果**: 更好地捕捉长序列的全局模式

## 文件结构

```
models/
  └── T3Time_FreTS_Gated_Qwen_LongSeq.py    # 改进模型文件

train_frets_gated_qwen_longseq.py           # 训练脚本

scripts/T3Time_FreTS_FusionExp/
  ├── train_longseq_multi_predlen.sh        # 批量训练脚本
  ├── analyze_longseq_results.py             # 结果分析脚本
  └── LONGSEQ_README.md                      # 本文档

test_frets_longseq_model.py                  # 快速测试脚本
```

## 使用方法

### 1. 快速测试模型

```bash
python test_frets_longseq_model.py
```

验证模型能否正常初始化和前向传播（支持 96, 192, 336, 720 预测长度）。

### 2. 单次训练

```bash
python train_frets_gated_qwen_longseq.py \
    --data_path ETTh1 \
    --seq_len 96 \
    --pred_len 192 \
    --channel 64 \
    --dropout_n 0.1 \
    --head 8 \
    --seed 2088 \
    --horizon_norm div1000 \
    --fusion_mode weighted_avg \
    --use_dynamic_sparsity 1 \
    --model_id T3Time_FreTS_Gated_Qwen_LongSeq_test
```

### 3. 批量训练（多个预测长度）

```bash
bash scripts/T3Time_FreTS_FusionExp/train_longseq_multi_predlen.sh
```

该脚本会在 96, 192, 336, 720 四个预测长度上分别训练，使用最佳配置：
- `horizon_norm=div1000`
- `fusion_mode=weighted_avg`
- `use_dynamic_sparsity=1`

### 4. 分析实验结果

```bash
python scripts/T3Time_FreTS_FusionExp/analyze_longseq_results.py
```

分析结果包括：
- 按预测长度的性能统计
- 按配置的性能统计
- 最佳结果展示
- 各预测长度的最佳配置

## 主要参数说明

### 模型参数

- `horizon_norm`: Horizon 归一化方式
  - `div1000`: 推荐，`pred_len / 1000.0`
  - `div100`: 原始方式，`pred_len / 100.0`
  - `log`: 对数归一化，`log(pred_len / 96.0)`

- `fusion_mode`: 融合模式
  - `weighted_avg`: 推荐，加权平均 `gate * freq + (1-gate) * time`
  - `residual`: 残差式 `time + gate * freq`

- `use_dynamic_sparsity`: 是否使用动态稀疏化
  - `1`: 启用，根据预测长度调整阈值
  - `0`: 禁用，使用固定阈值

### 训练参数

与原版训练脚本相同，主要参数：
- `--channel`: 通道数（默认 64）
- `--dropout_n`: Dropout 比率（默认 0.1）
- `--head`: 注意力头数（默认 8）
- `--learning_rate`: 学习率（默认 0.0001）
- `--seed`: 随机种子

## 预期改进效果

基于理论分析，预期改进：

| 预测长度 | 原版 MSE | 改进版 MSE (预期) | 改进幅度 |
|---------|---------|-----------------|---------|
| 96      | 0.370849 | ~0.370849      | 持平    |
| 192     | 较差     | 显著改善        | 10-20%  |
| 336     | 较差     | 显著改善        | 15-25%  |
| 720     | 较差     | 显著改善        | 20-30%  |

**注意**: 实际效果需要实验验证。

## 与原版对比

| 特性 | 原版 | 改进版 |
|------|------|--------|
| Horizon归一化 | `pred_len / 100.0` | `pred_len / 1000.0` |
| 融合公式 | `time + gate * freq` | `gate * freq + (1-gate) * time` |
| 稀疏化 | 固定阈值 | 动态调整 |
| 门控机制 | 局部特征 | 全局上下文 |
| 长序列性能 | 较差 | 预期改善 |

## 实验建议

1. **先测试短序列 (96)**: 确保改进版本在短序列上性能不下降
2. **逐步测试长序列**: 192 → 336 → 720
3. **对比不同配置**: 可以测试不同的 `horizon_norm` 和 `fusion_mode` 组合
4. **多种子验证**: 使用多个随机种子验证稳定性

## 故障排除

### 问题1: 模型初始化失败
- 检查 CUDA 是否可用
- 检查依赖包是否正确安装

### 问题2: 训练时内存不足
- 减小 `batch_size`
- 减小 `channel` 或 `head`

### 问题3: 结果没有改善
- 检查是否正确使用了改进配置
- 尝试不同的 `horizon_norm` 和 `fusion_mode` 组合
- 调整 `sparsity_threshold` 基础值

## 联系与反馈

如有问题或建议，请查看实验日志或联系开发者。
