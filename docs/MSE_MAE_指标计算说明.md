# MSE和MAE指标计算详解 - 多变量时间序列预测

## 1. 代码位置

- **文件**: `utils/metrics.py`
- **函数**: `MSE(pred, true)` 和 `MAE(pred, true)`
- **调用位置**: `train_frets_gated_qwen.py` 和 `train_frets_gated_qwen_fusion_exp.py` 的测试阶段

## 2. 代码实现

```python
def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))

def MSE(pred, true):
    return torch.mean((pred - true) ** 2)

def metric(pred, true):
    mse = MSE(pred, true).item()
    mae = MAE(pred, true).item()
    return mse, mae
```

## 3. 数据形状

在多变量时间序列预测任务中：

- **pred shape**: `(B, T, V)`
- **true shape**: `(B, T, V)`

其中：
- **B** = `batch_size` (批次大小，如16)
- **T** = `pred_len` (预测长度/时间步数，如96/192/336/720)
- **V** = `num_nodes` (变量数/特征维度，如7，对应ETTh1的7个变量)

## 4. MSE (Mean Squared Error) 详细计算

### 4.1 计算步骤

1. **计算逐元素误差**
   ```python
   error = pred - true
   # shape: (B, T, V)
   ```

2. **计算平方误差**
   ```python
   squared_error = (pred - true) ** 2
   # shape: (B, T, V)
   ```

3. **对所有维度求平均**
   ```python
   MSE = torch.mean(squared_error)
   # = mean over all dimensions (B, T, V)
   ```

### 4.2 数学公式

#### 完整形式：

$$
\text{MSE} = \frac{1}{B \times T \times V} \sum_{i=1}^{B} \sum_{j=1}^{T} \sum_{k=1}^{V} (\text{pred}_{i,j,k} - \text{true}_{i,j,k})^2
$$


#### 展开形式：

$$
\text{MSE} = \frac{1}{B \times T \times V} \times \left[
\begin{aligned}
&(\text{pred}_{1,1,1} - \text{true}_{1,1,1})^2 + (\text{pred}_{1,1,2} - \text{true}_{1,1,2})^2 + \cdots + (\text{pred}_{1,1,V} - \text{true}_{1,1,V})^2 + \\
&(\text{pred}_{1,2,1} - \text{true}_{1,2,1})^2 + (\text{pred}_{1,2,2} - \text{true}_{1,2,2})^2 + \cdots + (\text{pred}_{1,2,V} - \text{true}_{1,2,V})^2 + \\
&\vdots \\
&(\text{pred}_{1,T,1} - \text{true}_{1,T,1})^2 + (\text{pred}_{1,T,2} - \text{true}_{1,T,2})^2 + \cdots + (\text{pred}_{1,T,V} - \text{true}_{1,T,V})^2 + \\
&\vdots \\
&(\text{pred}_{B,T,1} - \text{true}_{B,T,1})^2 + (\text{pred}_{B,T,2} - \text{true}_{B,T,2})^2 + \cdots + (\text{pred}_{B,T,V} - \text{true}_{B,T,V})^2
\end{aligned}
\right]
$$

#### 简化形式（使用期望符号）：

$$
\text{MSE} = \mathbb{E}[(\text{pred} - \text{true})^2]
$$

其中 $\mathbb{E}[\cdot]$ 表示对所有 $(B, T, V)$ 维度的期望（平均）。

### 4.3 指标特点

- ✅ 对所有样本、所有时间步、所有变量的**平方误差**求平均
- ✅ 对大误差更敏感（平方放大了大误差）
- ✅ 单位：与原始数据单位的**平方**相同
- ✅ 最终得到一个**标量值**，表示整体预测性能

## 5. MAE (Mean Absolute Error) 详细计算

### 5.1 计算步骤

1. **计算逐元素误差**
   ```python
   error = pred - true
   # shape: (B, T, V)
   ```

2. **计算绝对误差**
   ```python
   abs_error = torch.abs(pred - true)
   # shape: (B, T, V)
   ```

3. **对所有维度求平均**
   ```python
   MAE = torch.mean(abs_error)
   # = mean over all dimensions (B, T, V)
   ```

### 5.2 数学公式

#### 完整形式：

$$
\text{MAE} = \frac{1}{B \times T \times V} \sum_{i=1}^{B} \sum_{j=1}^{T} \sum_{k=1}^{V} |\text{pred}_{i,j,k} - \text{true}_{i,j,k}|
$$

#### 展开形式：

$$
\text{MAE} = \frac{1}{B \times T \times V} \times \left[
\begin{aligned}
&|\text{pred}_{1,1,1} - \text{true}_{1,1,1}| + |\text{pred}_{1,1,2} - \text{true}_{1,1,2}| + \cdots + |\text{pred}_{1,1,V} - \text{true}_{1,1,V}| + \\
&|\text{pred}_{1,2,1} - \text{true}_{1,2,1}| + |\text{pred}_{1,2,2} - \text{true}_{1,2,2}| + \cdots + |\text{pred}_{1,2,V} - \text{true}_{1,2,V}| + \\
&\vdots \\
&|\text{pred}_{1,T,1} - \text{true}_{1,T,1}| + |\text{pred}_{1,T,2} - \text{true}_{1,T,2}| + \cdots + |\text{pred}_{1,T,V} - \text{true}_{1,T,V}| + \\
&\vdots \\
&|\text{pred}_{B,T,1} - \text{true}_{B,T,1}| + |\text{pred}_{B,T,2} - \text{true}_{B,T,2}| + \cdots + |\text{pred}_{B,T,V} - \text{true}_{B,T,V}|
\end{aligned}
\right]
$$

#### 简化形式（使用期望符号）：

$$
\text{MAE} = \mathbb{E}[|\text{pred} - \text{true}|]
$$

其中 $\mathbb{E}[\cdot]$ 表示对所有 $(B, T, V)$ 维度的期望（平均）。

### 5.3 指标特点

- ✅ 对所有样本、所有时间步、所有变量的**绝对误差**求平均
- ✅ 对所有误差**同等对待**（不放大任何误差）
- ✅ 单位：与原始数据单位**相同**
- ✅ 最终得到一个**标量值**，表示整体预测性能

## 6. 实际计算流程（从训练脚本）

### 6.1 测试阶段收集数据

```python
# 在 train_frets_gated_qwen.py 或 train_frets_gated_qwen_fusion_exp.py 中
model.eval()
preds, trues = [], []
with torch.no_grad():
    for i, batch_data in enumerate(test_loader):
        bx, by, bee = batch_data[0].to(device).float(), \
                      batch_data[1].to(device).float(), \
                      batch_data[-1].to(device).float()
        out = model(bx, None, bee)  # shape: (batch_size, pred_len, num_nodes)
        by_pred = by[:, -args.pred_len:, :]  # shape: (batch_size, pred_len, num_nodes)
        preds.append(out.detach().cpu())
        trues.append(by_pred.detach().cpu())
```

### 6.2 拼接所有batch

```python
preds = torch.cat(preds, dim=0)  # shape: (total_samples, pred_len, num_nodes)
trues = torch.cat(trues, dim=0)  # shape: (total_samples, pred_len, num_nodes)
```

其中 `total_samples = batch_size × num_batches`

### 6.3 计算指标

```python
mse, mae = metric(preds, trues)
print(f"On average horizons, Test MSE: {mse:.6f}, Test MAE: {mae:.6f}")
```

## 7. 示例计算

### 7.1 假设参数

- `batch_size = 16`
- `pred_len = 96`
- `num_nodes = 7` (ETTh1数据集有7个变量)
- 测试集有 `100` 个batch

### 7.2 数据形状

- `total_samples = 16 × 100 = 1600`
- `preds shape: (1600, 96, 7)`
- `trues shape: (1600, 96, 7)`
- `total_elements = 1600 × 96 × 7 = 1,075,200`

### 7.3 计算过程

$$
\text{MSE} = \frac{1}{1,075,200} \times \sum_{i=1}^{1600} \sum_{j=1}^{96} \sum_{k=1}^{7} (\text{pred}_{i,j,k} - \text{true}_{i,j,k})^2
$$

$$
\text{MAE} = \frac{1}{1,075,200} \times \sum_{i=1}^{1600} \sum_{j=1}^{96} \sum_{k=1}^{7} |\text{pred}_{i,j,k} - \text{true}_{i,j,k}|
$$

## 8. 在多变量时间序列中的特点

### 8.1 综合评估

- ✅ **同时考虑所有变量的预测误差**（不区分哪个变量）
- ✅ **同时考虑所有时间步的预测误差**（不区分哪个时间步）
- ✅ **同时考虑所有测试样本的预测误差**（不区分哪个样本）
- ✅ 最终得到一个**综合的误差指标**

### 8.2 不区分维度

- ❌ **不区分**哪个变量的误差更大
- ❌ **不区分**哪个时间步的误差更大
- ❌ **不区分**哪个样本的误差更大
- ✅ 所有维度**平等对待**，统一平均

### 8.3 适用场景

这种计算方式适用于：
- 多变量时间序列预测任务
- 需要整体性能评估的场景
- 所有变量同等重要的场景

如果需要区分不同变量或时间步的重要性，需要使用加权平均或其他评估方式。

## 9. MSE vs MAE 对比

| 特性 | MSE | MAE |
|------|-----|-----|
| **计算方式** | 平方误差的平均 | 绝对误差的平均 |
| **对大误差的敏感性** | 更敏感（平方放大） | 不敏感（线性） |
| **单位** | 原始单位的平方 | 与原始单位相同 |
| **数学性质** | 可导，便于优化 | 不可导（在0处） |
| **适用场景** | 需要惩罚大误差 | 需要稳健的评估 |

## 10. 总结

1. **MSE** 和 **MAE** 都是对**所有维度**（batch、时间步、变量）的误差进行**统一平均**
2. 两者都是**标量值**，表示整体预测性能
3. **MSE** 对大误差更敏感，**MAE** 对所有误差同等对待
4. 在多变量时间序列预测中，两者都提供了**综合的误差评估**
