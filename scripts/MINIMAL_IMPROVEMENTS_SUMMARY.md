# FreTS FusionExp 最小改动改进总结

## ✅ 已实施的改进（遵守奥卡姆剃刀原则）

### 改进 1: 损失函数对齐
**改动**: 从 `SmoothL1Loss` 改为 `MSELoss`（与 T3Time V30 一致）

**位置**: `train_frets_gated_qwen_fusion_exp.py`
- 添加 `--loss_fn` 参数，默认使用 `mse`
- 保持向后兼容，仍支持 `smooth_l1`

**理由**: T3Time V30 使用纯 MSE Loss，这可能是性能差异的原因之一

### 改进 2: 学习率调度器对齐
**改动**: 确保使用 `adjust_learning_rate` (Step Decay, type1)

**位置**: `train_frets_gated_qwen_fusion_exp.py`
- 已确认调用 `adjust_learning_rate(optimizer, epoch + 1, args)`
- 添加 `--lradj` 参数，默认 `type1`

**理由**: T3Time V30 使用 Step Decay，而不是 Cosine Annealing

### 改进 3: 归一化对齐
**改动**: 从 `affine=True` 改为 `affine=False`（与 T3Time V30 一致）

**位置**: `models/T3Time_FreTS_Gated_Qwen_FusionExp.py`
```python
# 从
self.normalize_layers = Normalize(num_nodes, affine=True).to(device)
# 改为
self.normalize_layers = Normalize(num_nodes, affine=False).to(device)
```

**理由**: T3Time V30 使用 `affine=False`，归一化方式可能影响特征分布

### 改进 4: FreTS Component 参数调整
**改动**: 降低稀疏化阈值（从 0.01 改为 0.005）

**位置**: `models/T3Time_FreTS_Gated_Qwen_FusionExp.py`
```python
# 从
def __init__(self, channel, seq_len, sparsity_threshold=0.01, ...):
# 改为
def __init__(self, channel, seq_len, sparsity_threshold=0.005, ...):
```

**理由**: 降低稀疏化阈值可以减少信息丢失，保留更多频域细节

## 🚀 使用方法

### 基础训练（使用所有改进）
```bash
python train_frets_gated_qwen_fusion_exp.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --channel 128 \
  --dropout_n 0.5 \
  --weight_decay 1e-3 \
  --fusion_mode gate \
  --loss_fn mse \
  --lradj type1 \
  --seed 2024 \
  --epochs 100
```

### 对比实验（测试损失函数影响）
```bash
# 使用 MSE Loss
python train_frets_gated_qwen_fusion_exp.py --loss_fn mse ...

# 使用 SmoothL1 Loss（原始）
python train_frets_gated_qwen_fusion_exp.py --loss_fn smooth_l1 ...
```

## 📊 预期改进

这些最小改动预期能够：
1. **对齐训练策略**: 与 T3Time V30 使用相同的损失函数和学习率调度
2. **减少信息丢失**: 降低稀疏化阈值，保留更多频域信息
3. **稳定特征分布**: 使用相同的归一化方式

**预期结果**: MSE 应该能够降低到 0.38 以下，接近或超越 T3Time V30 的性能

## 🔬 进一步优化（如果还不够）

如果上述改动还不够，可以考虑：

### 选项 A: 调整 FreTS Component 的 scale
```python
# 在模型定义中尝试不同的 scale 值
# scale=0.01, 0.02, 0.05
```

### 选项 B: 微调 horizon_info 权重
```python
# 当前: horizon_info = pred_len / 100.0
# 可以尝试: pred_len / 50.0 或 pred_len / 200.0
```

### 选项 C: 调整门控网络结构
```python
# 当前: channel // 2 -> channel
# 可以尝试: channel -> channel 或 channel // 4 -> channel
```

但这些都需要实验验证，建议先测试上述最小改动。

## 📝 注意事项

1. **归一化改动**: `affine=False` 意味着 RevIN 不会学习缩放和偏移参数，这与 T3Time V30 一致
2. **稀疏化阈值**: 降低阈值可能会增加计算量，但应该能保留更多信息
3. **损失函数**: MSE Loss 对大误差更敏感，可能需要更多训练轮次才能收敛

## 🎯 下一步

1. 运行改进后的训练脚本
2. 对比结果，看是否达到预期
3. 如果还不够，再考虑选项 A/B/C
