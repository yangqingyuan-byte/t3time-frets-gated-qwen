# FreDF 即插即用模块

这是一个独立的、即插即用的 FreDF 频率域损失函数模块，可以直接复制到您的项目中使用。

## 📦 文件说明

- `fredf_loss.py` - 核心损失函数模块（必需）
- `example.py` - 使用示例代码
- `QUICK_START.md` - 快速开始指南
- `INTEGRATION_GUIDE_CN.md` - 完整集成指南（中文）

## 🚀 快速开始

### 方法1：一行代码集成（最简单）

```python
import torch

# 在您的训练循环中
loss_temporal = ((outputs - batch_y) ** 2).mean()  # 原来的损失
loss_frequency = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()  # FreDF
loss = loss_temporal + 0.5 * loss_frequency  # 组合损失
```

### 方法2：使用提供的模块（推荐）

```python
from fredf_loss import FreDFLoss

# 创建损失函数
criterion = FreDFLoss(lambda_freq=0.5, loss_type='MSE')

# 在训练中使用
loss, loss_temporal, loss_frequency = criterion(outputs, batch_y)
```

### 方法3：使用简化函数

```python
from fredf_loss import simple_fredf_loss

# 一行代码
loss = simple_fredf_loss(outputs, batch_y, lambda_freq=0.5)
```

## 📋 使用步骤

1. **复制文件到您的项目**
   ```bash
   # 将整个 plug_and_play 文件夹复制到您的项目中
   cp -r plug_and_play /path/to/your/project/
   ```

2. **导入模块**
   ```python
   from fredf_loss import FreDFLoss
   # 或者
   from plug_and_play.fredf_loss import FreDFLoss
   ```

3. **在训练代码中使用**
   ```python
   criterion = FreDFLoss(lambda_freq=0.5)
   loss, loss_tmp, loss_freq = criterion(pred, target)
   ```

## 💡 完整示例

查看 `example.py` 文件获取完整的使用示例，包括：
- 最简单的集成方式
- 集成到训练循环
- 替换现有代码
- 完整训练循环示例

运行示例：
```bash
python example.py
```

## ⚙️ 参数说明

### FreDFLoss 参数

- `lambda_freq` (float): 频率损失权重，默认 0.5
  - 推荐范围：0.1 - 1.0
  - 起始值：0.5

- `loss_type` (str): 时域损失类型，默认 'MSE'
  - 'MSE': 均方误差
  - 'MAE': 平均绝对误差

- `freq_mode` (str): 频率变换模式，默认 'rfft'
  - 'rfft': 实值FFT（推荐，效率高）
  - 'fft': 完整FFT

- `auxi_type` (str): 频率损失类型，默认 'complex'
  - 'complex': 复数差异
  - 'mag': 仅幅度
  - 'phase': 仅相位
  - 'mag-phase': 幅度和相位

## 📖 详细文档

- [快速开始指南](QUICK_START.md)
- [完整集成指南（中文）](INTEGRATION_GUIDE_CN.md)
- [完整集成指南（English）](INTEGRATION_GUIDE.md)

## 🔧 依赖要求

- Python 3.8+
- PyTorch 1.8+

无需其他依赖！只需要 PyTorch 即可。

## ✅ 验证安装

运行以下代码验证模块是否正常工作：

```python
import torch
from fredf_loss import FreDFLoss

# 创建测试数据
pred = torch.randn(32, 96, 7)
target = torch.randn(32, 96, 7)

# 测试损失函数
criterion = FreDFLoss(lambda_freq=0.5)
loss, loss_tmp, loss_freq = criterion(pred, target)

print(f"总损失: {loss.item():.6f}")
print(f"时域损失: {loss_tmp.item():.6f}")
print(f"频域损失: {loss_freq.item():.6f}")
```

## ❓ 常见问题

**Q: 需要修改我的模型架构吗？**
A: 不需要！FreDF 是损失函数层面的改进，对模型架构没有任何要求。

**Q: 会影响训练速度吗？**
A: 影响很小。FFT 计算非常快，通常不会显著影响训练速度。

**Q: 适用于哪些任务？**
A: 适用于所有时间序列预测任务，包括长期预测、短期预测、插值等。

**Q: 如何调整参数？**
A: 建议从 `lambda_freq=0.5` 开始，根据验证集性能调整。范围通常在 0.1-1.0 之间。

## 📚 引用

如果您在研究中使用了 FreDF，请引用：

```bibtex
@inproceedings{wang2025fredf,
    title = {FreDF: Learning to Forecast in the Frequency Domain},
    author = {Wang, Hao and Pan, Licheng and Chen, Zhichao and Yang, Degui and Zhang, Sen and Yang, Yifei and Liu, Xinggao and Li, Haoxuan and Tao, Dacheng},
    booktitle = {ICLR},
    year = {2025},
}
```

## 📝 许可证

本项目遵循 MIT 许可证。

