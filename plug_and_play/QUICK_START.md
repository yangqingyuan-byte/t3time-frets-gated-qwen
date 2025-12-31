# FreDF 快速开始

## 一行代码集成

```python
import torch

# 在您的训练循环中，替换原来的损失计算
loss_temporal = ((outputs - batch_y) ** 2).mean()  # 原来的损失
loss_frequency = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()  # FreDF
loss = loss_temporal + 0.5 * loss_frequency  # 组合损失
```

## 使用提供的模块

```python
from fredf_loss import FreDFLoss

criterion = FreDFLoss(lambda_freq=0.5, loss_type='MSE')
loss, loss_tmp, loss_freq = criterion(outputs, batch_y)
```

## 完整示例

```python
import torch
import torch.nn as nn
from fredf_loss import FreDFLoss

# 您的模型
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = FreDFLoss(lambda_freq=0.5)

# 训练循环
for batch_x, batch_y in train_loader:
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss, _, _ = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
```

## 参数说明

- `lambda_freq`: 频率损失权重（推荐：0.5）
- `loss_type`: 时域损失类型（'MSE' 或 'MAE'）
- `freq_mode`: 频率模式（'rfft' 推荐，或 'fft'）

## 详细文档

- [完整集成指南（中文）](INTEGRATION_GUIDE_CN.md)
- [完整集成指南（English）](INTEGRATION_GUIDE.md)
- [示例代码](examples/integration_example.py)

