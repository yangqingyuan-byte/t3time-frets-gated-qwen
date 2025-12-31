"""
FreDF 集成示例
演示如何将 FreDF 频率域损失函数集成到您自己的模型中

使用方法：
    python example.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 导入 FreDF 损失函数
from fredf_loss import FreDFLoss, simple_fredf_loss


# ==================== 示例1: 最简单的集成方式 ====================
def example_simple_integration():
    """最简单的集成方式 - 一行代码"""
    print("=" * 60)
    print("示例1: 最简单的集成方式")
    print("=" * 60)
    
    # 模拟数据
    batch_size, pred_len, num_features = 32, 96, 7
    outputs = torch.randn(batch_size, pred_len, num_features)
    batch_y = torch.randn(batch_size, pred_len, num_features)
    
    # 原始方式（仅时域损失）
    loss_original = ((outputs - batch_y) ** 2).mean()
    print(f"原始损失（仅时域）: {loss_original.item():.6f}")
    
    # FreDF 方式（时域 + 频域）
    loss_fredf = simple_fredf_loss(outputs, batch_y, lambda_freq=0.5)
    print(f"FreDF 损失（时域+频域）: {loss_fredf.item():.6f}")
    print()


# ==================== 示例2: 集成到训练循环 ====================
def example_training_loop():
    """集成到完整的训练循环"""
    print("=" * 60)
    print("示例2: 集成到训练循环")
    print("=" * 60)
    
    # 创建一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.linear(x)
    
    # 初始化模型和优化器
    model = SimpleModel(input_dim=96, output_dim=96)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建 FreDF 损失函数
    criterion = FreDFLoss(lambda_freq=0.5, loss_type='MSE', freq_mode='rfft')
    
    # 模拟训练数据
    batch_x = torch.randn(32, 96, 7)
    batch_y = torch.randn(32, 96, 7)
    
    # 训练步骤
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(batch_x)
    
    # 计算损失
    loss, loss_temporal, loss_frequency = criterion(outputs, batch_y)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    print(f"总损失: {loss.item():.6f}")
    print(f"  时域损失: {loss_temporal.item():.6f}")
    print(f"  频域损失: {loss_frequency.item():.6f}")
    print()


# ==================== 示例3: 替换现有代码 ====================
def example_replace_existing():
    """替换现有代码中的损失函数"""
    print("=" * 60)
    print("示例3: 替换现有代码")
    print("=" * 60)
    
    # 假设这是您原来的训练代码
    def original_training_step(model, batch_x, batch_y, criterion):
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)  # 原来的损失
        return loss
    
    # 修改后的代码（添加 FreDF）
    def fredf_training_step(model, batch_x, batch_y, criterion_temporal):
        outputs = model(batch_x)
        
        # 原来的时域损失
        loss_temporal = criterion_temporal(outputs, batch_y)
        
        # 添加频率域损失（FreDF 核心）
        loss_frequency = (torch.fft.rfft(outputs, dim=1) - 
                         torch.fft.rfft(batch_y, dim=1)).abs().mean()
        
        # 组合损失
        loss = loss_temporal + 0.5 * loss_frequency
        return loss, loss_temporal, loss_frequency
    
    # 使用示例
    model = nn.Linear(96, 96)
    batch_x = torch.randn(32, 96, 7)
    batch_y = torch.randn(32, 96, 7)
    criterion = nn.MSELoss()
    
    # 原来的方式
    loss_original = original_training_step(model, batch_x, batch_y, criterion)
    print(f"原始损失: {loss_original.item():.6f}")
    
    # FreDF 方式
    loss_fredf, loss_tmp, loss_freq = fredf_training_step(model, batch_x, batch_y, criterion)
    print(f"FreDF 损失: {loss_fredf.item():.6f}")
    print(f"  时域: {loss_tmp.item():.6f}, 频域: {loss_freq.item():.6f}")
    print()


# ==================== 示例4: 完整训练循环 ====================
def example_full_training():
    """完整的训练循环示例"""
    print("=" * 60)
    print("示例4: 完整训练循环")
    print("=" * 60)
    
    class TimeSeriesModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=7, hidden_size=64, num_layers=2, batch_first=True)
            self.fc = nn.Linear(64, 7)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out)
    
    # 创建模型和数据
    model = TimeSeriesModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = FreDFLoss(lambda_freq=0.5, loss_type='MSE')
    
    # 模拟数据加载器
    train_data = torch.randn(100, 96, 7)
    train_labels = torch.randn(100, 96, 7)
    dataset = TensorDataset(train_data, train_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 训练几个epoch
    num_epochs = 2
    for epoch in range(num_epochs):
        total_loss = 0
        total_tmp = 0
        total_freq = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(batch_x)
            
            # 计算损失
            loss, loss_tmp, loss_freq = criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_tmp += loss_tmp.item()
            total_freq += loss_freq.item()
        
        num_batches = len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  平均总损失: {total_loss/num_batches:.6f}")
        print(f"  平均时域损失: {total_tmp/num_batches:.6f}")
        print(f"  平均频域损失: {total_freq/num_batches:.6f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FreDF 集成示例")
    print("=" * 60 + "\n")
    
    # 运行所有示例
    example_simple_integration()
    example_training_loop()
    example_replace_existing()
    example_full_training()
    
    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)

