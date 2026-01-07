# T3Time_FreTS_Gated_Qwen 模型调试指南

## 📊 问题概述

根据实验日志，`T3Time_FreTS_Gated_Qwen` 模型的表现明显不如 `T3Time_Pro_Qwen_SOTA_V30`：
- **FreTS**: MSE ~0.38-0.39, MAE ~0.39-0.50
- **T3Time V30**: MSE ~0.38, MAE ~0.41

## 🔍 核心差异分析

### 1. 频域处理方式

**T3Time (V30)**:
- 使用 **Learnable Wavelet Packet Decomposition** (可学习小波包分解)
- 多频带分解 (4个节点: Node 0-3)
- 频带间交互 (Cross-Frequency Interaction)
- 先验引导初始化 (Prior-Guided Init)

**FreTS**:
- 使用 **FreTS Component** (频域 MLP)
- 单频域表示
- FFT → MLP → IFFT
- SoftShrink 稀疏化

### 2. 架构差异

| 组件 | T3Time V30 | FreTS |
|------|-----------|-------|
| 频域分支 | Wavelet Packet (4 bands) | FreTS (single) |
| 融合方式 | Static Weights + Horizon Gate | Cross-Attention |
| Channel | 128 | 64 |
| Dropout | 0.5 | 0.1 |
| Weight Decay | 1e-3 | 1e-4 |

## 🛠️ 系统性调试方案

### 阶段一：基础诊断

#### 1.1 运行诊断脚本

```bash
python scripts/debug_frets_model.py
```

该脚本会检查：
- 模型参数量对比
- 输出差异分析
- 特征统计信息
- 融合机制诊断
- 频域分支分析

#### 1.2 检查训练过程

```bash
# 训练并监控
python train_frets_gated_qwen.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --channel 64 \
  --dropout_n 0.1 \
  --weight_decay 1e-4 \
  --seed 2024 \
  --epochs 10  # 先用少量 epoch 测试
```

**观察点**：
- Train Loss 是否正常下降
- Val Loss 是否过拟合
- 梯度是否正常（无 NaN/Inf）

### 阶段二：组件级诊断

#### 2.1 频域分支诊断

**问题假设**：FreTS Component 可能存在问题

**诊断步骤**：

1. **检查 FreTS 输出范围**
```python
# 在模型 forward 中添加调试代码
def forward(self, input_data, input_data_mark, embeddings):
    # ... 前面的代码 ...
    
    fre_processed = self.frets_branch(fre_input)
    
    # 【调试】检查频域输出
    print(f"FreTS 输出统计: mean={fre_processed.mean():.4f}, std={fre_processed.std():.4f}")
    print(f"FreTS 输出范围: [{fre_processed.min():.4f}, {fre_processed.max():.4f}]")
    
    # ... 后续代码 ...
```

2. **检查频域特征是否退化**
```python
# 检查频域特征是否接近零或过大
if fre_processed.abs().mean() < 1e-6:
    print("⚠️ 警告: 频域特征接近零，可能退化")
if fre_processed.abs().mean() > 100:
    print("⚠️ 警告: 频域特征过大，可能爆炸")
```

#### 2.2 融合机制诊断

**问题假设**：Cross-Attention 融合可能失效

**诊断步骤**：

1. **检查注意力权重分布**
```python
# 修改 cross_attn_fusion 调用
fused_attn, attn_weights = self.cross_attn_fusion(
    time_encoded, fre_encoded, fre_encoded, 
    average_attn_weights=False  # 返回注意力权重
)
print(f"注意力权重统计: mean={attn_weights.mean():.4f}, std={attn_weights.std():.4f}")
```

2. **检查融合后的特征**
```python
fused_features = self.fusion_norm(fused_attn + time_encoded)
print(f"融合特征 vs 时域特征: ratio={fused_features.norm() / time_encoded.norm():.4f}")
# 如果 ratio 接近 1，说明频域信息没有融入
```

#### 2.3 归一化诊断

**问题假设**：RevIN 归一化可能有问题

**诊断步骤**：

```python
# 检查归一化前后的统计
x_norm = self.normalize_layers(x, 'norm')
print(f"归一化前: mean={x.mean():.4f}, std={x.std():.4f}")
print(f"归一化后: mean={x_norm.mean():.4f}, std={x_norm.std():.4f}")

# 检查反归一化
out = self.normalize_layers(dec_out, 'denorm')
print(f"反归一化后: mean={out.mean():.4f}, std={out.std():.4f}")
```

### 阶段三：对比实验

#### 3.1 渐进式替换实验

**实验 1**: 替换频域分支
```python
# 在 FreTS 模型中使用 Wavelet Packet 替代 FreTS Component
# 修改 models/T3Time_FreTS_Gated_Qwen.py
# 将 self.frets_branch 替换为 Wavelet Packet 处理
```

**实验 2**: 替换融合方式
```python
# 使用 T3Time 的静态权重融合替代 Cross-Attention
# 添加 band_weights 和 fusion_gate
```

**实验 3**: 调整超参数
```python
# 尝试 T3Time 的超参数配置
--channel 128
--dropout_n 0.5
--weight_decay 1e-3
```

#### 3.2 消融实验

创建消融实验脚本：

```bash
# 实验 A: 移除频域分支
python train_frets_gated_qwen_ablation.py --ablation no_freq

# 实验 B: 移除 Cross-Attention，使用简单相加
python train_frets_gated_qwen_ablation.py --ablation simple_fusion

# 实验 C: 移除 CMA，使用简单 Decoder
python train_frets_gated_qwen_ablation.py --ablation no_cma
```

### 阶段四：深度分析

#### 4.1 梯度分析

```python
# 在训练循环中添加
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 100:
            print(f"⚠️ 梯度爆炸: {name}, norm={grad_norm:.4f}")
        if grad_norm < 1e-6:
            print(f"⚠️ 梯度消失: {name}, norm={grad_norm:.4f}")
```

#### 4.2 损失函数分析

```python
# 分别计算各部分的损失
mse_loss = criterion(outputs, batch_y_pred)
print(f"MSE Loss: {mse_loss.item():.6f}")

# 检查预测分布
print(f"预测统计: mean={outputs.mean():.4f}, std={outputs.std():.4f}")
print(f"真实统计: mean={batch_y_pred.mean():.4f}, std={batch_y_pred.std():.4f}")
```

## 🎯 可能的根本原因

### 1. FreTS Component 设计问题

**症状**：
- 频域特征退化或爆炸
- 频域信息没有有效利用

**解决方案**：
- 调整 `sparsity_threshold`
- 修改 FreTS 的 MLP 结构
- 增加频域特征的归一化

### 2. 融合机制失效

**症状**：
- Cross-Attention 权重分布异常
- 融合后特征与原始特征差异小

**解决方案**：
- 尝试不同的融合方式（加权求和、门控等）
- 调整融合层的初始化
- 增加融合层的深度

### 3. 超参数不匹配

**症状**：
- 训练不稳定
- 过拟合或欠拟合

**解决方案**：
- 参考 T3Time 的超参数配置
- 进行超参数网格搜索
- 使用学习率调度器

### 4. 归一化问题

**症状**：
- 输出分布异常
- 训练后期性能下降

**解决方案**：
- 检查 RevIN 的 affine 参数
- 尝试不同的归一化策略
- 调整归一化的位置

## 📝 调试检查清单

- [ ] 运行基础诊断脚本
- [ ] 检查训练日志（Loss 曲线）
- [ ] 验证频域分支输出
- [ ] 验证融合机制
- [ ] 对比 T3Time 和 FreTS 的输出
- [ ] 进行消融实验
- [ ] 调整超参数
- [ ] 分析梯度流
- [ ] 检查归一化
- [ ] 验证数据加载

## 🔧 快速修复建议

### 优先级 1: 超参数对齐

```bash
python train_frets_gated_qwen.py \
  --channel 128 \
  --dropout_n 0.5 \
  --weight_decay 1e-3 \
  --learning_rate 5e-5
```

### 优先级 2: 频域分支优化

考虑：
1. 增加 FreTS 的容量
2. 调整稀疏化阈值
3. 添加频域特征的残差连接

### 优先级 3: 融合机制改进

考虑：
1. 使用多尺度融合
2. 添加门控机制
3. 引入频域权重

## 📚 参考资源

- T3Time V30 模型代码: `models/T3Time_Learnable_Wavelet_Packet_Gated_Pro_Qwen.py`
- FreTS 原始论文实现
- 诊断脚本: `scripts/debug_frets_model.py`
