# FreDF 集成指南

本指南说明如何将 FreDF 的频率域损失函数集成到您自己的时间序列预测模型中。

## 核心思想

FreDF 通过在频率域添加辅助损失函数来增强时间序列预测模型的性能。核心思想是：
- **时域损失**：传统的 MSE/MAE 损失
- **频域损失**：在频率域比较预测值和真实值的差异

## 快速集成（最简单方式）

### 方法1：一行代码集成（推荐）

在您的训练循环中，将原来的损失函数替换为：

```python
import torch

# 假设 outputs 是模型预测值，batch_y 是真实值
# 形状: [batch_size, pred_len, num_features]

# 原始时域损失
loss_temporal = ((outputs - batch_y) ** 2).mean()  # MSE
# 或者
loss_temporal = (outputs - batch_y).abs().mean()  # MAE

# FreDF 频率域损失（核心代码）
loss_frequency = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()

# 组合损失（可调整权重）
lambda_freq = 0.5  # 频率损失权重，可根据任务调整
loss = loss_temporal + lambda_freq * loss_frequency
```

### 方法2：封装为函数

创建一个可复用的损失函数：

```python
import torch
import torch.nn as nn

class FreDFLoss(nn.Module):
    """
    FreDF 频率域损失函数
    
    参数:
        lambda_freq: 频率损失权重，默认 0.5
        loss_type: 时域损失类型，'MSE' 或 'MAE'
        freq_mode: 频率变换模式，'rfft' (推荐) 或 'fft'
    """
    def __init__(self, lambda_freq=0.5, loss_type='MSE', freq_mode='rfft'):
        super().__init__()
        self.lambda_freq = lambda_freq
        self.loss_type = loss_type
        self.freq_mode = freq_mode
        
        if loss_type == 'MSE':
            self.temporal_loss = nn.MSELoss()
        elif loss_type == 'MAE':
            self.temporal_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
    
    def forward(self, pred, target, dim=1):
        """
        计算组合损失
        
        参数:
            pred: 预测值，形状 [B, T, D] 或 [B, T]
            target: 真实值，形状 [B, T, D] 或 [B, T]
            dim: 时间维度，默认 1
        """
        # 时域损失
        loss_temporal = self.temporal_loss(pred, target)
        
        # 频域损失
        if self.freq_mode == 'rfft':
            # 实值FFT（推荐，计算效率更高）
            freq_pred = torch.fft.rfft(pred, dim=dim)
            freq_target = torch.fft.rfft(target, dim=dim)
        elif self.freq_mode == 'fft':
            # 完整FFT
            freq_pred = torch.fft.fft(pred, dim=dim)
            freq_target = torch.fft.fft(target, dim=dim)
        else:
            raise ValueError(f"Unsupported freq_mode: {freq_mode}")
        
        # 计算频率域差异的绝对值平均
        loss_frequency = (freq_pred - freq_target).abs().mean()
        
        # 组合损失
        loss = loss_temporal + self.lambda_freq * loss_frequency
        
        return loss, loss_temporal, loss_frequency

# 使用示例
criterion = FreDFLoss(lambda_freq=0.5, loss_type='MSE', freq_mode='rfft')
loss, loss_tmp, loss_freq = criterion(outputs, batch_y)
```

## 完整集成示例

### 示例1：集成到 PyTorch 训练循环

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 假设您有一个模型
class YourModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 您的模型定义
        pass
    
    def forward(self, x):
        # 您的模型前向传播
        return predictions

# 初始化模型和优化器
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 创建 FreDF 损失函数
lambda_freq = 0.5  # 可调整的超参数
criterion_temporal = nn.MSELoss()

# 训练循环
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        
        # 模型预测
        outputs = model(batch_x)
        
        # 计算时域损失
        loss_temporal = criterion_temporal(outputs, batch_y)
        
        # 计算频域损失（FreDF 核心）
        loss_frequency = (torch.fft.rfft(outputs, dim=1) - 
                         torch.fft.rfft(batch_y, dim=1)).abs().mean()
        
        # 组合损失
        loss = loss_temporal + lambda_freq * loss_frequency
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 可选：打印损失
        if step % 100 == 0:
            print(f'Epoch {epoch}, Step {step}')
            print(f'  Temporal Loss: {loss_temporal.item():.4f}')
            print(f'  Frequency Loss: {loss_frequency.item():.4f}')
            print(f'  Total Loss: {loss.item():.4f}')
```

### 示例2：集成到现有项目（最小改动）

如果您已经有训练代码，只需要修改损失计算部分：

```python
# 原来的代码
loss = criterion(outputs, batch_y)

# 修改为（添加频率损失）
loss_temporal = criterion(outputs, batch_y)
loss_frequency = (torch.fft.rfft(outputs, dim=1) - 
                 torch.fft.rfft(batch_y, dim=1)).abs().mean()
loss = loss_temporal + 0.5 * loss_frequency  # 0.5 是可调权重
```

## 高级用法

### 1. 不同频率域损失类型

FreDF 支持多种频率域损失计算方式：

```python
# 方式1：实值FFT（推荐，效率高）
loss_freq = (torch.fft.rfft(outputs, dim=1) - 
             torch.fft.rfft(batch_y, dim=1)).abs().mean()

# 方式2：完整FFT
loss_freq = (torch.fft.fft(outputs, dim=1) - 
             torch.fft.fft(batch_y, dim=1)).abs().mean()

# 方式3：仅幅度（magnitude）
freq_pred = torch.fft.rfft(outputs, dim=1).abs()
freq_target = torch.fft.rfft(batch_y, dim=1).abs()
loss_freq = (freq_pred - freq_target).abs().mean()

# 方式4：仅相位（phase）
freq_pred = torch.fft.rfft(outputs, dim=1).angle()
freq_target = torch.fft.rfft(batch_y, dim=1).angle()
loss_freq = (freq_pred - freq_target).abs().mean()

# 方式5：MSE 形式的频率损失
loss_freq = ((torch.fft.rfft(outputs, dim=1) - 
              torch.fft.rfft(batch_y, dim=1)).abs() ** 2).mean()
```

### 2. 自适应权重调整

可以根据训练进度动态调整频率损失权重：

```python
# 线性衰减
lambda_freq = max(0.5 * (1 - epoch / num_epochs), 0.1)

# 余弦退火
lambda_freq = 0.5 * (1 + np.cos(np.pi * epoch / num_epochs))

# 固定权重（最简单）
lambda_freq = 0.5
```

### 3. 多维度频率损失

如果您的数据是多变量的，可以在特征维度也计算频率损失：

```python
# 时间维度（默认）
loss_freq_time = (torch.fft.rfft(outputs, dim=1) - 
                  torch.fft.rfft(batch_y, dim=1)).abs().mean()

# 特征维度
loss_freq_feat = (torch.fft.rfft(outputs, dim=-1) - 
                  torch.fft.rfft(batch_y, dim=-1)).abs().mean()

# 组合
loss = loss_temporal + lambda_freq * (loss_freq_time + loss_freq_feat)
```

## 参数调优建议

1. **lambda_freq（频率损失权重）**
   - 推荐范围：0.1 - 1.0
   - 起始值：0.5
   - 如果频率损失过大，可以降低；如果效果不明显，可以增加

2. **freq_mode（频率模式）**
   - `rfft`：推荐，计算效率高，适用于实值时间序列
   - `fft`：完整FFT，适用于复数时间序列

3. **loss_type（时域损失类型）**
   - `MSE`：均方误差，适合大多数情况
   - `MAE`：平均绝对误差，对异常值更鲁棒

## 注意事项

1. **维度要求**
   - 确保 `outputs` 和 `batch_y` 的形状一致
   - `dim=1` 通常对应时间维度 `[batch, time, features]`

2. **设备一致性**
   - 确保所有张量在同一设备（CPU/GPU）上

3. **梯度计算**
   - `torch.fft.rfft` 和 `torch.fft.fft` 支持自动求导，可以直接用于训练

4. **内存使用**
   - 频率域损失会增加一定的内存开销，如果内存不足，可以考虑：
     - 使用 `rfft` 而不是 `fft`
     - 减小批次大小
     - 使用梯度累积

## 验证集成是否成功

添加以下代码来验证频率损失是否正常工作：

```python
# 检查频率损失的数值范围
print(f"Temporal loss: {loss_temporal.item():.6f}")
print(f"Frequency loss: {loss_frequency.item():.6f}")
print(f"Total loss: {loss.item():.6f}")

# 检查梯度是否存在
print(f"Model parameter gradient exists: {model.parameters().__next__().grad is not None}")
```

## 参考实现

项目中的完整实现可以参考：
- `exp/exp_long_term_forecasting.py`：长期预测任务
- `exp/exp_short_term_forecasting.py`：短期预测任务
- `exp/exp_imputation.py`：插值任务

## 常见问题

**Q: 频率损失应该设置多大权重？**
A: 建议从 0.5 开始，根据验证集性能调整。通常范围在 0.1-1.0 之间。

**Q: 可以使用频率损失单独训练吗？**
A: 可以，但通常与时域损失组合效果更好。可以尝试 `loss = loss_frequency` 看看效果。

**Q: 对模型架构有要求吗？**
A: 没有特殊要求，任何时间序列预测模型都可以使用。

**Q: 会影响训练速度吗？**
A: 频率变换计算开销很小，通常不会显著影响训练速度。

## 引用

如果您在研究中使用了 FreDF，请引用：

```bibtex
@inproceedings{wang2025fredf,
    title = {FreDF: Learning to Forecast in the Frequency Domain},
    author = {Wang, Hao and Pan, Licheng and Chen, Zhichao and Yang, Degui and Zhang, Sen and Yang, Yifei and Liu, Xinggao and Li, Haoxuan and Tao, Dacheng},
    booktitle = {ICLR},
    year = {2025},
}
```

