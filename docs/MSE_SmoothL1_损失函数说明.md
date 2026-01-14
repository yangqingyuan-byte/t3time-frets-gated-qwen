# MSE和SmoothL1损失函数详解 - 多变量时间序列预测

## 1. 代码位置

- **文件**: `train_frets_gated_qwen.py` 和 `train_frets_gated_qwen_fusion_exp.py`
- **定义位置**: 第99-102行（`train_frets_gated_qwen.py`）
- **使用位置**: 
  - 训练阶段：第124行 `loss = criterion(outputs, batch_y_pred)`
  - 验证阶段：第136行 `vali_loss.append(criterion(out, by_pred).item())`

## 2. 代码实现

```python
# 损失函数选择
if args.loss_fn == 'mse':
    criterion = nn.MSELoss()
else:
    criterion = nn.SmoothL1Loss(beta=0.2)

# 训练时使用
loss = criterion(outputs, batch_y_pred)
```

## 3. 数据形状

在多变量时间序列预测任务中：

- **outputs shape**: `(B, T, V)` - 模型预测值
- **batch_y_pred shape**: `(B, T, V)` - 真实目标值

其中：
- **B** = `batch_size` (批次大小，如16)
- **T** = `pred_len` (预测长度/时间步数，如96/192/336/720)
- **V** = `num_nodes` (变量数/特征维度，如7，对应ETTh1的7个变量)

## 4. MSE Loss (Mean Squared Error Loss)

### 4.1 PyTorch实现

```python
criterion = nn.MSELoss()
loss = criterion(pred, target)
```

**默认参数**:
- `reduction='mean'`: 对所有元素求平均
- 无其他参数

### 4.2 数学公式

#### 对于单个元素：

设预测值为 `pred`，真实值为 `target`，则单个元素的损失为：

$$
\text{loss}_{\text{element}} = (\text{pred} - \text{target})^2
$$

#### 对于整个张量 (shape: B×T×V)：

$$
\text{MSE\_Loss} = \frac{1}{B \times T \times V} \sum_{i=1}^{B} \sum_{j=1}^{T} \sum_{k=1}^{V} (\text{pred}_{i,j,k} - \text{target}_{i,j,k})^2
$$

#### 展开形式：

$$
\text{MSE\_Loss} = \frac{1}{B \times T \times V} \times \left[
\begin{aligned}
&(\text{pred}_{1,1,1} - \text{target}_{1,1,1})^2 + (\text{pred}_{1,1,2} - \text{target}_{1,1,2})^2 + \cdots + (\text{pred}_{1,1,V} - \text{target}_{1,1,V})^2 + \\
&(\text{pred}_{1,2,1} - \text{target}_{1,2,1})^2 + (\text{pred}_{1,2,2} - \text{target}_{1,2,2})^2 + \cdots + (\text{pred}_{1,2,V} - \text{target}_{1,2,V})^2 + \\
&\vdots \\
&(\text{pred}_{1,T,1} - \text{target}_{1,T,1})^2 + (\text{pred}_{1,T,2} - \text{target}_{1,T,2})^2 + \cdots + (\text{pred}_{1,T,V} - \text{target}_{1,T,V})^2 + \\
&\vdots \\
&(\text{pred}_{B,T,1} - \text{target}_{B,T,1})^2 + (\text{pred}_{B,T,2} - \text{target}_{B,T,2})^2 + \cdots + (\text{pred}_{B,T,V} - \text{target}_{B,T,V})^2
\end{aligned}
\right]
$$

#### 简化形式（使用期望符号）：

$$
\text{MSE\_Loss} = \mathbb{E}[(\text{pred} - \text{target})^2]
$$

其中 $\mathbb{E}[\cdot]$ 表示对所有 $(B, T, V)$ 维度的期望（平均）。

### 4.3 梯度计算

对于反向传播，需要计算损失对预测值的梯度：

$$
\frac{\partial \text{MSE\_Loss}}{\partial \text{pred}_{i,j,k}} = \frac{2}{B \times T \times V} (\text{pred}_{i,j,k} - \text{target}_{i,j,k})
$$

### 4.4 特点

- ✅ **对所有元素统一平均**：同时考虑所有batch、所有时间步、所有变量
- ✅ **对大误差敏感**：平方运算放大了大误差的影响
- ✅ **可导性**：处处可导，梯度稳定
- ✅ **单位**：与原始数据单位的**平方**相同
- ✅ **优化特性**：梯度与误差成正比，大误差时梯度也大

## 5. SmoothL1 Loss (Smooth L1 Loss / Huber Loss)

### 5.1 PyTorch实现

```python
criterion = nn.SmoothL1Loss(beta=0.2)
loss = criterion(pred, target)
```

**参数说明**:
- `beta=0.2`: 控制平滑区域的大小（阈值参数）
- `reduction='mean'`: 对所有元素求平均（默认）

### 5.2 数学公式

#### 对于单个元素：

设：
- $x = \text{pred} - \text{target}$ (误差)
- $\beta = 0.2$ (beta参数，平滑阈值)

则单个元素的损失为分段函数：

$$
\text{loss}_{\text{element}} = \begin{cases}
\frac{1}{2} \cdot \frac{x^2}{\beta} & \text{if } |x| < \beta \\
|x| - \frac{1}{2} \beta & \text{if } |x| \geq \beta
\end{cases}
$$

#### beta=0.2时的具体公式：

当 $\beta = 0.2$ 时：

$$
\text{loss}_{\text{element}} = \begin{cases}
2.5 \cdot x^2 & \text{if } |x| < 0.2 \\
|x| - 0.1 & \text{if } |x| \geq 0.2
\end{cases}
$$

**推导过程**：
- 当 $|x| < 0.2$ 时：$\text{loss}_{\text{element}} = \frac{1}{2} \cdot \frac{x^2}{0.2} = 2.5 \cdot x^2$
- 当 $|x| \geq 0.2$ 时：$\text{loss}_{\text{element}} = |x| - \frac{1}{2} \cdot 0.2 = |x| - 0.1$

#### 对于整个张量 (shape: B×T×V)：

$$
\text{SmoothL1\_Loss} = \frac{1}{B \times T \times V} \sum_{i=1}^{B} \sum_{j=1}^{T} \sum_{k=1}^{V} \text{loss}_{\text{element}}(\text{pred}_{i,j,k} - \text{target}_{i,j,k})
$$

其中 $\text{loss}_{\text{element}}(\cdot)$ 是上述分段函数。

#### 简化形式（使用期望符号）：

$$
\text{SmoothL1\_Loss} = \mathbb{E}[\text{loss}_{\text{element}}(\text{pred} - \text{target})]
$$

其中 $\mathbb{E}[\cdot]$ 表示对所有 $(B, T, V)$ 维度的期望（平均）。

### 5.3 梯度计算

对于反向传播，需要计算损失对预测值的梯度：

$$
\frac{\partial \text{SmoothL1\_Loss}}{\partial \text{pred}_{i,j,k}} = \begin{cases}
\frac{1}{B \times T \times V} \cdot \frac{x}{\beta} = \frac{1}{B \times T \times V} \cdot 5x & \text{if } |x| < \beta \\
\frac{1}{B \times T \times V} \cdot \text{sign}(x) & \text{if } |x| \geq \beta
\end{cases}
$$

其中 $x = \text{pred}_{i,j,k} - \text{target}_{i,j,k}$，$\text{sign}(x)$ 是符号函数。

**beta=0.2时的梯度**：
- 当 $|x| < 0.2$ 时：梯度 = $\frac{1}{B \times T \times V} \cdot 5x$（线性，与误差成正比）
- 当 $|x| \geq 0.2$ 时：梯度 = $\frac{1}{B \times T \times V} \cdot \text{sign}(x)$（常数，不随误差大小变化）

### 5.4 函数图像特性

SmoothL1 Loss 结合了 MSE 和 MAE 的优点：

1. **小误差区域** ($|x| < \beta$):
   - 行为类似 MSE（平方损失）
   - 对小误差敏感，有助于精细调整
   - 梯度与误差成正比，梯度连续

2. **大误差区域** ($|x| \geq \beta$):
   - 行为类似 MAE（线性损失）
   - 对大误差稳健，避免异常值过度影响
   - 梯度为常数，防止梯度爆炸

### 5.5 特点

- ✅ **结合MSE和MAE优点**：小误差用平方，大误差用线性
- ✅ **对小误差敏感**：在平滑区域内，类似MSE的平方行为
- ✅ **对大误差稳健**：在大误差区域，梯度不随误差增大而增大
- ✅ **可导性**：处处可导，梯度稳定
- ✅ **beta参数**：控制平滑区域的大小，beta越小，平滑区域越小

## 6. 实际计算流程

### 6.1 训练阶段

```python
# 在 train_frets_gated_qwen.py 中
model.train()
for i, batch_data in enumerate(data_loader):
    optimizer.zero_grad()
    batch_x, batch_y, be = batch_data[0].to(device).float(), \
                           batch_data[1].to(device).float(), \
                           batch_data[-1].to(device).float()
    
    # 模型前向传播
    outputs = model(batch_x, None, be)  # shape: (B, T, V)
    
    # 提取目标值（最后pred_len个时间步）
    batch_y_pred = batch_y[:, -args.pred_len:, :]  # shape: (B, T, V)
    
    # 计算损失
    loss = criterion(outputs, batch_y_pred)  # 标量值
    
    # 反向传播
    loss.backward()
    optimizer.step()
```

### 6.2 验证阶段

```python
model.eval()
vali_loss = []
with torch.no_grad():
    for i, batch_data in enumerate(vali_loader):
        bx, by, bee = batch_data[0].to(device).float(), \
                     batch_data[1].to(device).float(), \
                     batch_data[-1].to(device).float()
        out = model(bx, None, bee)
        by_pred = by[:, -args.pred_len:, :]
        vali_loss.append(criterion(out, by_pred).item())

avg_vali_loss = np.mean(vali_loss)
```

## 7. 示例计算

### 7.1 假设参数

- `batch_size = 16`
- `pred_len = 96`
- `num_nodes = 7`
- `beta = 0.2` (SmoothL1 Loss)

### 7.2 数据形状

- `outputs shape: (16, 96, 7)`
- `batch_y_pred shape: (16, 96, 7)`
- `total_elements = 16 × 96 × 7 = 10,752`

### 7.3 MSE Loss计算

$$
\text{MSE\_Loss} = \frac{1}{10,752} \sum_{i=1}^{16} \sum_{j=1}^{96} \sum_{k=1}^{7} (\text{outputs}_{i,j,k} - \text{batch\_y\_pred}_{i,j,k})^2
$$

### 7.4 SmoothL1 Loss计算

对于每个元素 $x_{i,j,k} = \text{outputs}_{i,j,k} - \text{batch\_y\_pred}_{i,j,k}$：

$$
\text{SmoothL1\_Loss} = \frac{1}{10,752} \sum_{i=1}^{16} \sum_{j=1}^{96} \sum_{k=1}^{7} \begin{cases}
2.5 \cdot x_{i,j,k}^2 & \text{if } |x_{i,j,k}| < 0.2 \\
|x_{i,j,k}| - 0.1 & \text{if } |x_{i,j,k}| \geq 0.2
\end{cases}
$$

## 8. MSE vs SmoothL1 对比

| 特性 | MSE Loss | SmoothL1 Loss (β=0.2) |
|------|----------|------------------------|
| **小误差行为** | 平方 $(x^2)$ | 平方 $(2.5x^2)$ |
| **大误差行为** | 平方（放大）$(x^2)$ | 线性 $(|x|-0.1)$ |
| **对大误差的敏感性** | 高（平方放大） | 低（稳健） |
| **梯度特性** | 与误差成正比 | 小误差时与误差成正比，大误差时为常数 |
| **梯度稳定性** | 好 | 更好（大误差时梯度不爆炸） |
| **可导性** | 是 | 是 |
| **单位** | 原始单位的平方 | 与原始单位相同（大误差区域） |
| **适用场景** | 需要惩罚大误差 | 需要稳健训练，避免异常值影响 |

## 9. 在多变量时间序列中的特点

### 9.1 统一计算

- ✅ **同时考虑所有变量的预测误差**（不区分哪个变量）
- ✅ **同时考虑所有时间步的预测误差**（不区分哪个时间步）
- ✅ **同时考虑所有样本的预测误差**（不区分哪个样本）
- ✅ 最终得到一个**标量损失值**

### 9.2 训练中的作用

- **损失函数用于反向传播**：通过最小化损失函数来更新模型参数
- **梯度计算**：损失函数对预测值的梯度用于参数更新
- **优化目标**：损失值越小，模型预测越准确

### 9.3 与评估指标的区别

| 项目 | 损失函数 (Loss) | 评估指标 (Metric) |
|------|----------------|-------------------|
| **用途** | 训练时优化模型 | 测试时评估性能 |
| **计算时机** | 每个batch都计算 | 测试集全部预测完后计算 |
| **参与反向传播** | 是 | 否 |
| **示例** | MSE Loss, SmoothL1 Loss | MSE, MAE (在utils/metrics.py中) |

**注意**：虽然名称相同（如MSE），但损失函数和评估指标的计算方式是一致的，只是用途不同。

## 10. beta参数的影响

### 10.1 beta的作用

`beta` 参数控制 SmoothL1 Loss 的平滑区域大小：

- **beta 越小**：平滑区域越小，更接近 MAE
- **beta 越大**：平滑区域越大，更接近 MSE

### 10.2 不同beta值的比较

| beta值 | 平滑区域 | 行为特点 |
|--------|---------|---------|
| 0.1 | $|x| < 0.1$ | 更接近MAE，对大误差更稳健 |
| **0.2** | $|x| < 0.2$ | **项目默认值**，平衡MSE和MAE |
| 0.5 | $|x| < 0.5$ | 更接近MSE，对小误差更敏感 |
| 1.0 | $|x| < 1.0$ | 非常接近MSE |

### 10.3 选择beta的建议

- **数据中有异常值**：选择较小的beta（如0.1-0.2），提高稳健性
- **数据质量好**：选择较大的beta（如0.5-1.0），提高对小误差的敏感性
- **项目默认**：beta=0.2，平衡稳健性和敏感性

## 11. 总结

1. **MSE Loss** 和 **SmoothL1 Loss** 都是对**所有维度**（batch、时间步、变量）的误差进行**统一平均**

2. **MSE Loss**：
   - 对所有误差使用平方
   - 对大误差敏感
   - 梯度与误差成正比

3. **SmoothL1 Loss**：
   - 小误差用平方（类似MSE），大误差用线性（类似MAE）
   - 结合了MSE和MAE的优点
   - 对大误差稳健，梯度不爆炸

4. 两者都用于**训练过程**，通过反向传播优化模型参数

5. 在多变量时间序列预测中，两者都提供了**综合的损失评估**，不区分变量、时间步或样本
