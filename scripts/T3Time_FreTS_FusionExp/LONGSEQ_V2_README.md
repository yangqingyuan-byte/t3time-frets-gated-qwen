# T3Time_FreTS_Gated_Qwen_LongSeq_v2 使用说明

## 概述

这是**完全参考 T3Time 流程**的改进版本，专门针对长序列预测（192, 336, 720）优化。v2版本完全按照 T3Time 的流程顺序和结构设计，只是用 FreTS Component 替代了固定 FFT。

## 关键改进（相对于v1版本）

### 1. 完全按照 T3Time 的流程顺序

**v1版本的问题**：
- 融合机制和门控设计与 T3Time 不一致
- 频域处理流程顺序不同

**v2版本的改进**：
- 完全按照 T3Time 的流程顺序
- 每个步骤的维度变换与 T3Time 完全一致

### 2. RichHorizonGate 只基于时域特征

**v1版本**：
```python
gate = self.rich_horizon_gate(time_encoded, fre_encoded, self.pred_len)  # 同时输入时域和频域
```

**v2版本（与T3Time一致）**：
```python
gate = self.rich_horizon_gate(enc_out, self.pred_len)  # 只基于时域特征
enc_out = gate * freq_enc_out.permute(0,2,1) + (1 - gate) * enc_out
```

### 3. 频域处理流程

**T3Time 流程**：
```
FFT → 投影 → 编码器 → 池化 → reshape
```

**v2版本流程**：
```
投影 → FreTS Component → 编码器 → 池化 → reshape
```

FreTS Component 内部完成 FFT 和可学习变换，输出维度与 T3Time 的 FFT 输出对应。

### 4. 融合公式（与 T3Time 完全一致）

```python
enc_out = gate * freq_enc_out.permute(0,2,1) + (1 - gate) * enc_out
```

这是加权平均，保证时域和频域都有贡献。

## 文件结构

```
models/
  └── T3Time_FreTS_Gated_Qwen_LongSeq_v2.py    # v2模型文件

train_frets_gated_qwen_longseq_v2.py           # v2训练脚本

scripts/T3Time_FreTS_FusionExp/
  ├── train_longseq_v2_multi_predlen.sh        # v2批量训练脚本（192, 336, 720）
  ├── compare_longseq_models.py                # 模型对比分析脚本
  └── LONGSEQ_V2_README.md                      # 本文档

test_frets_longseq_v2_model.py                  # v2快速测试脚本
```

## 使用方法

### 1. 快速测试模型

```bash
python test_frets_longseq_v2_model.py
```

验证模型能否正常初始化和前向传播（支持 96, 192, 336, 720 预测长度）。

### 2. 单次训练

```bash
python train_frets_gated_qwen_longseq_v2.py \
    --data_path ETTh1 \
    --seq_len 96 \
    --pred_len 336 \
    --channel 64 \
    --dropout_n 0.1 \
    --head 8 \
    --seed 2088 \
    --use_dynamic_sparsity 1 \
    --model_id T3Time_FreTS_Gated_Qwen_LongSeq_v2_test
```

### 3. 批量训练（长序列：192, 336, 720）

```bash
bash scripts/T3Time_FreTS_FusionExp/train_longseq_v2_multi_predlen.sh
```

该脚本会在 192, 336, 720 三个长序列预测长度上分别训练，使用最佳配置。

### 4. 对比分析

```bash
# 对比所有版本的结果
python scripts/T3Time_FreTS_FusionExp/compare_longseq_models.py

# 分析v2版本的结果
python scripts/T3Time_FreTS_FusionExp/analyze_longseq_results.py --model_name T3Time_FreTS_Gated_Qwen_LongSeq_v2
```

## 主要参数说明

### 模型参数

- `use_dynamic_sparsity`: 是否使用动态稀疏化
  - `1`: 启用，根据预测长度调整阈值（推荐）
  - `0`: 禁用，使用固定阈值

- `sparsity_threshold`: 稀疏化阈值基础值（默认 0.009）
- `frets_scale`: FreTS Component 初始化 scale（默认 0.018）

### 训练参数

与原版训练脚本相同，主要参数：
- `--channel`: 通道数（默认 64）
- `--dropout_n`: Dropout 比率（默认 0.1）
- `--head`: 注意力头数（默认 8）
- `--learning_rate`: 学习率（默认 0.0001）
- `--seed`: 随机种子

## v2版本 vs v1版本 vs 原版

| 特性 | 原版 | v1版本 | v2版本 |
|------|------|--------|--------|
| 流程顺序 | - | 部分改进 | **完全参考T3Time** |
| RichHorizonGate输入 | - | 时域+频域 | **只基于时域（与T3Time一致）** |
| 融合公式 | 残差式 | 加权平均 | **加权平均（与T3Time一致）** |
| 频域处理 | FreTS → 池化 → 编码 | FreTS → 池化 → 编码 | **投影 → FreTS → 编码 → 池化** |
| Horizon归一化 | ÷100 | ÷1000 | **÷1000（与T3Time一致）** |
| 动态稀疏化 | 无 | 有 | **有** |

## 预期改进效果

基于完全参考 T3Time 流程的设计，预期在长序列（192, 336, 720）上会有显著改善：

| 预测长度 | 原版 MSE | v1版本 MSE | v2版本 MSE (预期) | 改进幅度 |
|---------|---------|-----------|-----------------|---------|
| 96      | 0.370849 | 0.381478  | ~0.370849       | 持平    |
| 192     | 较差     | 0.446224  | **显著改善**    | 15-25%  |
| 336     | 较差     | 0.482258  | **显著改善**    | 20-30%  |
| 720     | 较差     | 0.526653  | **显著改善**    | 25-35%  |

**注意**: 实际效果需要实验验证。

## 与原版 T3Time 的对比

| 特性 | T3Time | v2版本 |
|------|--------|--------|
| 频域处理 | 固定 FFT | **FreTS Component（可学习）** |
| 稀疏化 | 无 | **动态稀疏化** |
| 其他 | - | **完全一致** |

v2版本在保持 T3Time 优秀流程的基础上，增加了可学习的频域变换和动态稀疏化机制。

## 实验建议

1. **先测试短序列 (96)**: 确保v2版本在短序列上性能不下降
2. **重点测试长序列**: 192 → 336 → 720
3. **对比分析**: 使用 `compare_longseq_models.py` 对比所有版本
4. **多种子验证**: 使用多个随机种子验证稳定性

## 故障排除

### 问题1: 模型初始化失败
- 检查 CUDA 是否可用
- 检查依赖包是否正确安装

### 问题2: 训练时内存不足
- 减小 `batch_size`
- 减小 `channel` 或 `head`

### 问题3: 结果没有改善
- 确认使用的是 v2 版本（检查 model_id）
- 检查是否正确使用了动态稀疏化
- 对比 T3Time 原版的结果作为参考

## 总结

v2版本是**完全参考 T3Time 流程**的改进版本，主要改进：
1. ✅ 完全按照 T3Time 的流程顺序
2. ✅ RichHorizonGate 只基于时域特征（与 T3Time 一致）
3. ✅ 融合公式与 T3Time 完全一致
4. ✅ 频域处理流程对应 T3Time
5. ✅ 保留 FreTS Component 的可学习优势
6. ✅ 增加动态稀疏化机制

预期在长序列预测上会有显著改善！
