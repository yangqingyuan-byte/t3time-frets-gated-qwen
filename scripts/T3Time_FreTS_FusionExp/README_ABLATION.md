# T3Time_FreTS_FusionExp 消融实验说明

## 📋 模型改进说明

模型已更新以支持消融实验中的组件切换功能。

### 新增参数

模型现在支持以下参数来控制不同组件的使用：

1. **`use_frets`** (bool): 
   - `True`: 使用 FreTS Component（可学习频域 MLP）
   - `False`: 使用固定 FFT（原始 T3Time 方式）

2. **`use_complex`** (bool):
   - `True`: 使用复数信息
   - `False`: 仅使用幅度（仅当 `use_frets=False` 时有效）
   - 注意：当前实现中，固定 FFT 模式都使用幅度，此参数为未来扩展预留

3. **`use_sparsity`** (bool):
   - `True`: 使用稀疏化机制（SoftShrink）
   - `False`: 不使用稀疏化（仅当 `use_frets=True` 时有效）

4. **`use_improved_gate`** (bool):
   - `True`: 使用改进门控（基于归一化输入）
   - `False`: 使用原始门控（基于注意力输出）

## 🚀 使用方法

### 训练脚本参数

在训练脚本中，可以通过以下参数控制：

```bash
python train_frets_gated_qwen_fusion_exp.py \
  --use_frets 1 \          # 1=使用FreTS, 0=使用固定FFT
  --use_complex 1 \        # 1=使用复数, 0=仅幅度
  --use_sparsity 1 \       # 1=使用稀疏化, 0=不使用
  --use_improved_gate 1 \  # 1=改进门控, 0=原始门控
  ...
```

### 消融实验脚本

消融实验脚本会自动设置这些参数：

- **`ablation_study.sh`**: 完整消融实验（包含所有实验）
- **`ablation_study_simple.sh`**: 简化消融实验（只包含可直接运行的实验）
- **`quick_ablation_test.sh`**: 快速测试（用于验证脚本）

## 📊 消融实验配置

### 实验1: FreTS Component 的有效性

- **A1_FFT_Magnitude**: 固定 FFT + 仅幅度
- **A1_FreTS_Component**: FreTS Component + 稀疏化
- **A2_FFT_Magnitude_Only**: 固定 FFT + 仅幅度
- **A2_FFT_Complex**: 固定 FFT + 复数（当前实现仍使用幅度）
- **A3_FreTS_NoSparsity**: FreTS Component + 无稀疏化
- **A3_FreTS_WithSparsity**: FreTS Component + 稀疏化

### 实验2: 融合机制对比

- **A4_Fusion_Gate**: Gate 模式
- **A4_Fusion_Weighted**: Weighted 模式
- **A4_Fusion_CrossAttn**: Cross-Attention 模式
- **A4_Fusion_Hybrid**: Hybrid 模式

### 实验3: 超参数敏感性分析

- **A5_Scale_***: 测试不同的 scale 值
- **A6_Sparsity_***: 测试不同的 sparsity_threshold 值

### 实验4: 门控机制改进的影响

- **A7_Original_Gate**: 原始门控
- **A7_Improved_Gate**: 改进门控

## 🔧 模型代码改进

### 1. GatedTransformerEncoderLayer

支持 `use_improved_gate` 参数：
- `True`: 门控基于归一化输入（改进方式）
- `False`: 门控基于注意力输出（原始方式）

### 2. FreTSComponent

支持 `use_sparsity` 参数：
- `True`: 使用 SoftShrink 稀疏化
- `False`: 不使用稀疏化

### 3. TriModalFreTSGatedQwenFusionExp

支持完整的组件切换：
- 根据 `use_frets` 选择 FreTS Component 或固定 FFT
- 根据 `use_improved_gate` 选择门控方式
- 所有编码器层都支持门控方式切换

## ✅ 验证

模型已通过以下测试：
- ✅ FreTS 模式实例化
- ✅ 固定 FFT 模式实例化
- ✅ 原始门控模式实例化
- ✅ Forward 前向传播测试

## 📝 注意事项

1. **固定 FFT 模式**: 当前实现中，`use_complex` 参数在固定 FFT 模式下都使用幅度，此参数为未来扩展预留。

2. **参数依赖关系**:
   - `use_complex` 仅在 `use_frets=False` 时有效
   - `use_sparsity` 仅在 `use_frets=True` 时有效

3. **默认值**: 所有新参数的默认值都是 `True`，保持向后兼容。

## 🎯 运行消融实验

```bash
# 完整消融实验
bash scripts/T3Time_FreTS_FusionExp/ablation_study.sh

# 简化消融实验（推荐）
bash scripts/T3Time_FreTS_FusionExp/ablation_study_simple.sh

# 快速测试
bash scripts/T3Time_FreTS_FusionExp/quick_ablation_test.sh
```

## 📈 分析结果

实验完成后，运行分析脚本：

```bash
python scripts/T3Time_FreTS_FusionExp/analyze_ablation_results.py
```
