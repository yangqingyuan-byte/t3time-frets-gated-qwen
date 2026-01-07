# T3Time_FreEformer_Gated_Qwen 参数寻优顺序建议

## 📊 参数分类和影响分析

### 第一优先级：架构参数（影响模型容量和表达能力）

这些参数决定了模型的基本架构，影响最大，应该最先优化。

| 参数 | 影响范围 | 推荐搜索空间 | 说明 |
|------|---------|------------|------|
| **`channel`** | ⭐⭐⭐⭐⭐ | [32, 64, 96, 128] | 模型的核心维度，影响所有分支（时域、频域、融合） |
| **`fre_e_layer`** | ⭐⭐⭐⭐ | [1, 2, 3] | 频域 Transformer 编码器层数，直接影响频域处理能力 |
| **`embed_size`** | ⭐⭐⭐ | [8, 16, 32] | 频域 token embedding 维度，影响频域表示能力 |
| **`e_layer`** | ⭐⭐⭐ | [1, 2] | 时域编码器层数 |
| **`head`** | ⭐⭐ | [4, 8, 12, 16] | 注意力头数（需确保 channel % head == 0） |

**建议顺序**：
1. **先固定其他参数，只调整 `channel`**（32, 64, 96, 128）
2. **固定最佳 `channel`，调整 `fre_e_layer`**（1, 2, 3）
3. **固定前两者，调整 `embed_size`**（8, 16, 32）
4. **固定前三个，微调 `e_layer` 和 `head`**

### 第二优先级：训练参数（影响训练稳定性和收敛）

在确定架构后，优化训练过程。

| 参数 | 影响范围 | 推荐搜索空间 | 说明 |
|------|---------|------------|------|
| **`learning_rate`** | ⭐⭐⭐⭐⭐ | [5e-5, 7.5e-5, 1e-4, 1.5e-4] | 学习率，影响收敛速度和最终性能 |
| **`dropout_n`** | ⭐⭐⭐⭐ | [0.1, 0.2, 0.3, 0.4, 0.5] | Dropout 比率，防止过拟合 |
| **`batch_size`** | ⭐⭐⭐ | [16, 32, 64] | 批次大小，影响训练稳定性和内存占用 |
| **`weight_decay`** | ⭐⭐ | [1e-4, 5e-4, 1e-3] | 权重衰减，L2 正则化强度 |

**建议顺序**：
1. **先调整 `learning_rate`**（固定其他训练参数）
2. **固定最佳 `learning_rate`，调整 `dropout_n`**
3. **固定前两者，调整 `batch_size`**
4. **最后微调 `weight_decay`**

### 第三优先级：注意力机制参数（细粒度优化）

这些参数影响频域注意力机制的具体行为，在架构和训练参数确定后再优化。

| 参数 | 影响范围 | 推荐搜索空间 | 说明 |
|------|---------|------------|------|
| **`attn_enhance`** | ⭐⭐⭐ | [None/0, 1] | 注意力增强模式（Vanilla vs Enhanced） |
| **`attn_softmax_flag`** | ⭐⭐ | [0, 1] | 是否使用 softmax 归一化权重矩阵 |
| **`attn_weight_plus`** | ⭐⭐ | [0, 1] | 权重加法模式（乘法 vs 加法） |
| **`attn_outside_softmax`** | ⭐ | [0, 1] | 是否在 softmax 外部应用权重矩阵 |

**建议顺序**：
1. **先测试 `attn_enhance`**（0 vs 1，决定是否使用增强注意力）
2. **如果 `attn_enhance=1`，再调整其他三个参数**
3. **如果 `attn_enhance=0`，可以跳过其他三个参数**

### 第四优先级：其他参数（微调）

| 参数 | 影响范围 | 推荐搜索空间 | 说明 |
|------|---------|------------|------|
| **`loss_fn`** | ⭐⭐ | ['mse', 'smooth_l1'] | 损失函数 |
| **`d_layer`** | ⭐ | [1, 2] | 解码器层数（通常 1 层足够） |

## 🎯 推荐的寻优流程

### 阶段 1：架构参数寻优（最重要）

```bash
# 步骤 1.1: 只调整 channel（固定其他参数）
channel: [32, 64, 96, 128]
fre_e_layer: 1 (固定)
embed_size: 16 (固定)
e_layer: 1 (固定)
head: 8 (固定)
learning_rate: 1e-4 (固定)
dropout_n: 0.1 (固定)
batch_size: 32 (固定)
```

**预期实验数**: 4 个

---

```bash
# 步骤 1.2: 固定最佳 channel，调整 fre_e_layer
channel: <最佳值> (固定)
fre_e_layer: [1, 2, 3]
embed_size: 16 (固定)
e_layer: 1 (固定)
head: 8 (固定)
learning_rate: 1e-4 (固定)
dropout_n: 0.1 (固定)
batch_size: 32 (固定)
```

**预期实验数**: 3 个

---

```bash
# 步骤 1.3: 固定前两者，调整 embed_size
channel: <最佳值> (固定)
fre_e_layer: <最佳值> (固定)
embed_size: [8, 16, 32]
e_layer: 1 (固定)
head: 8 (固定)
learning_rate: 1e-4 (固定)
dropout_n: 0.1 (固定)
batch_size: 32 (固定)
```

**预期实验数**: 3 个

---

### 阶段 2：训练参数寻优

```bash
# 步骤 2.1: 调整 learning_rate（固定架构参数）
channel: <阶段1最佳值> (固定)
fre_e_layer: <阶段1最佳值> (固定)
embed_size: <阶段1最佳值> (固定)
learning_rate: [5e-5, 7.5e-5, 1e-4, 1.5e-4]
dropout_n: 0.1 (固定)
batch_size: 32 (固定)
```

**预期实验数**: 4 个

---

```bash
# 步骤 2.2: 固定最佳 learning_rate，调整 dropout_n
learning_rate: <阶段2.1最佳值> (固定)
dropout_n: [0.1, 0.2, 0.3, 0.4, 0.5]
batch_size: 32 (固定)
```

**预期实验数**: 5 个

---

```bash
# 步骤 2.3: 固定前两者，调整 batch_size
learning_rate: <阶段2.1最佳值> (固定)
dropout_n: <阶段2.2最佳值> (固定)
batch_size: [16, 32, 64]
```

**预期实验数**: 3 个

---

### 阶段 3：注意力机制参数寻优（可选）

```bash
# 步骤 3.1: 测试 attn_enhance
attn_enhance: [None/0, 1]
attn_softmax_flag: 1 (固定)
attn_weight_plus: 0 (固定)
attn_outside_softmax: 0 (固定)
```

**预期实验数**: 2 个

---

```bash
# 步骤 3.2: 如果 attn_enhance=1，调整其他参数
attn_enhance: 1 (固定)
attn_softmax_flag: [0, 1]
attn_weight_plus: [0, 1]
attn_outside_softmax: [0, 1]
```

**预期实验数**: 2 × 2 × 2 = 8 个（如果 attn_enhance=1）

---

### 阶段 4：最终微调（可选）

```bash
# 测试 loss_fn 和 d_layer
loss_fn: ['mse', 'smooth_l1']
d_layer: [1, 2]
```

**预期实验数**: 2 × 2 = 4 个

---

## 📈 总实验数估算

- **阶段 1（架构参数）**: 4 + 3 + 3 = **10 个实验**
- **阶段 2（训练参数）**: 4 + 5 + 3 = **12 个实验**
- **阶段 3（注意力参数）**: 2 + 8 = **10 个实验**（如果 attn_enhance=1）
- **阶段 4（微调）**: **4 个实验**

**总计**: 约 **36 个实验**（如果跳过阶段 3 和 4，只需 **22 个实验**）

## 🚀 快速开始建议

### 最小化寻优（快速验证）

如果时间有限，只做阶段 1 和阶段 2：

1. **阶段 1**: 调整 `channel` 和 `fre_e_layer`（7 个实验）
2. **阶段 2**: 调整 `learning_rate` 和 `dropout_n`（9 个实验）

**总计**: **16 个实验**

### 完整寻优（推荐）

按照上述四个阶段完整执行，确保找到最佳参数组合。

## ⚠️ 注意事项

1. **参数依赖关系**:
   - `head` 必须满足 `channel % head == 0`
   - 如果 `channel=64`，`head` 可以是 [1, 2, 4, 8, 16, 32, 64]
   - 如果 `channel=128`，`head` 可以是 [1, 2, 4, 8, 16, 32, 64, 128]

2. **内存限制**:
   - `channel` 越大，模型参数量越大，内存占用越高
   - `batch_size` 越大，内存占用越高
   - 建议根据 GPU 内存调整

3. **训练时间**:
   - 每个实验建议至少训练 30-50 个 epoch
   - 使用 early stopping（patience=8-10）可以提前终止

4. **多种子验证**:
   - 找到最佳参数组合后，建议用多个种子（如 2020-2040）验证稳定性
   - 最终结果取多个种子的平均值

## 📝 示例脚本结构

```bash
# 阶段 1.1: channel 寻优
for channel in 32 64 96 128; do
    python train_freeformer_gated_qwen.py \
        --channel $channel \
        --fre_e_layer 1 \
        --embed_size 16 \
        --learning_rate 1e-4 \
        --dropout_n 0.1 \
        --batch_size 32 \
        ...
done

# 阶段 1.2: fre_e_layer 寻优（使用阶段1.1的最佳channel）
BEST_CHANNEL=64  # 从阶段1.1得到
for fre_e_layer in 1 2 3; do
    python train_freeformer_gated_qwen.py \
        --channel $BEST_CHANNEL \
        --fre_e_layer $fre_e_layer \
        ...
done
```

## 🎯 总结

**推荐寻优顺序**：
1. ✅ **架构参数**（channel → fre_e_layer → embed_size）
2. ✅ **训练参数**（learning_rate → dropout_n → batch_size）
3. ⚠️ **注意力参数**（attn_enhance → 其他三个，可选）
4. ⚠️ **微调参数**（loss_fn, d_layer，可选）

**关键原则**：
- 先粗调（架构），后细调（训练）
- 一次只调整一个参数（或少量相关参数）
- 固定其他参数，确保结果可比较
- 找到最佳值后再进入下一阶段
