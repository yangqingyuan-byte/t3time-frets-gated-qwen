"""
FreDF Loss Function Module
即插即用的频率域损失函数模块

使用方法:
    from fredf_loss import FreDFLoss
    
    criterion = FreDFLoss(lambda_freq=0.5, loss_type='MSE')
    loss, loss_tmp, loss_freq = criterion(pred, target)
"""

import torch
import torch.nn as nn


class FreDFLoss(nn.Module):
    """
    FreDF 频率域损失函数
    
    这是一个即插即用的损失函数，可以轻松集成到任何时间序列预测模型中。
    
    参数:
        lambda_freq (float): 频率损失权重，默认 0.5
        loss_type (str): 时域损失类型，'MSE' 或 'MAE'，默认 'MSE'
        freq_mode (str): 频率变换模式，'rfft' (推荐) 或 'fft'，默认 'rfft'
        auxi_type (str): 频率损失类型，'complex', 'mag', 'phase', 'mag-phase'，默认 'complex'
        module_first (bool): 是否先计算模再平均，默认 True
    
    示例:
        >>> criterion = FreDFLoss(lambda_freq=0.5, loss_type='MSE')
        >>> pred = torch.randn(32, 96, 7)  # [batch, time, features]
        >>> target = torch.randn(32, 96, 7)
        >>> loss, loss_tmp, loss_freq = criterion(pred, target)
    """
    
    def __init__(self, lambda_freq=0.5, loss_type='MSE', freq_mode='rfft', 
                 auxi_type='complex', module_first=True):
        super().__init__()
        self.lambda_freq = lambda_freq
        self.loss_type = loss_type
        self.freq_mode = freq_mode
        self.auxi_type = auxi_type
        self.module_first = module_first
        
        # 初始化时域损失函数
        if loss_type.upper() == 'MSE':
            self.temporal_loss = nn.MSELoss()
        elif loss_type.upper() == 'MAE':
            self.temporal_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Use 'MSE' or 'MAE'.")
        
        # 验证频率模式
        if freq_mode not in ['rfft', 'fft']:
            raise ValueError(f"Unsupported freq_mode: {freq_mode}. Use 'rfft' or 'fft'.")
        
        # 验证频率损失类型
        valid_auxi_types = ['complex', 'mag', 'phase', 'mag-phase']
        if auxi_type not in valid_auxi_types:
            raise ValueError(f"Unsupported auxi_type: {auxi_type}. Use {valid_auxi_types}.")
    
    def forward(self, pred, target, dim=1):
        """
        计算组合损失
        
        参数:
            pred (torch.Tensor): 预测值，形状 [B, T, D] 或 [B, T]
            target (torch.Tensor): 真实值，形状 [B, T, D] 或 [B, T]
            dim (int): 时间维度，默认 1
        
        返回:
            tuple: (总损失, 时域损失, 频域损失)
        """
        # 计算时域损失
        loss_temporal = self.temporal_loss(pred, target)
        
        # 计算频域损失
        loss_frequency = self._compute_frequency_loss(pred, target, dim)
        
        # 组合损失
        loss = loss_temporal + self.lambda_freq * loss_frequency
        
        return loss, loss_temporal, loss_frequency
    
    def _compute_frequency_loss(self, pred, target, dim):
        """
        计算频率域损失
        
        参数:
            pred: 预测值
            target: 真实值
            dim: 时间维度
        
        返回:
            频率域损失值
        """
        # 执行FFT变换
        if self.freq_mode == 'rfft':
            freq_pred = torch.fft.rfft(pred, dim=dim)
            freq_target = torch.fft.rfft(target, dim=dim)
        else:  # fft
            freq_pred = torch.fft.fft(pred, dim=dim)
            freq_target = torch.fft.fft(target, dim=dim)
        
        # 根据类型计算频率差异
        if self.auxi_type == 'complex':
            # 直接使用复数差异
            loss_auxi = freq_pred - freq_target
        elif self.auxi_type == 'mag':
            # 仅使用幅度
            loss_auxi = freq_pred.abs() - freq_target.abs()
        elif self.auxi_type == 'phase':
            # 仅使用相位
            loss_auxi = freq_pred.angle() - freq_target.angle()
        elif self.auxi_type == 'mag-phase':
            # 幅度和相位分别计算
            loss_auxi_mag = freq_pred.abs() - freq_target.abs()
            loss_auxi_phase = freq_pred.angle() - freq_target.angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        else:
            raise ValueError(f"Unsupported auxi_type: {self.auxi_type}")
        
        # 计算损失值
        if self.loss_type.upper() == 'MSE':
            if self.module_first:
                loss_freq = (loss_auxi.abs() ** 2).mean()
            else:
                loss_freq = (loss_auxi ** 2).mean().abs()
        else:  # MAE
            if self.module_first:
                loss_freq = loss_auxi.abs().mean()
            else:
                loss_freq = loss_auxi.mean().abs()
        
        return loss_freq


def simple_fredf_loss(pred, target, lambda_freq=0.5, dim=1):
    """
    简化版 FreDF 损失函数（一行代码版本）
    
    这是最简单的使用方式，适合快速集成。
    
    参数:
        pred (torch.Tensor): 预测值
        target (torch.Tensor): 真实值
        lambda_freq (float): 频率损失权重，默认 0.5
        dim (int): 时间维度，默认 1
    
    返回:
        torch.Tensor: 总损失值
    
    示例:
        >>> pred = torch.randn(32, 96, 7)
        >>> target = torch.randn(32, 96, 7)
        >>> loss = simple_fredf_loss(pred, target, lambda_freq=0.5)
    """
    # 时域损失（MSE）
    loss_temporal = ((pred - target) ** 2).mean()
    
    # 频域损失（FreDF 核心）
    loss_frequency = (torch.fft.rfft(pred, dim=dim) - 
                     torch.fft.rfft(target, dim=dim)).abs().mean()
    
    # 组合损失
    loss = loss_temporal + lambda_freq * loss_frequency
    
    return loss


# 为了向后兼容，也可以直接导入函数
__all__ = ['FreDFLoss', 'simple_fredf_loss']

