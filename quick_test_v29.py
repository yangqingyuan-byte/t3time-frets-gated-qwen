#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试 V29 模型的前向传播"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.T3Time_Learnable_Wavelet_Packet_Gated_Pro_Qwen import TriModalLearnableWaveletPacketGatedProQwen

def test_v29():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型实例（V29 配置）
    model = TriModalLearnableWaveletPacketGatedProQwen(
        device=device,
        channel=128,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.5,
        wp_level=2
    ).to(device)
    
    # 创建测试输入
    batch_size = 2
    x = torch.randn(batch_size, 96, 7).to(device)  # [B, L, N]
    x_mark = torch.randn(batch_size, 96, 4).to(device)  # [B, L, 4]
    emb = torch.randn(batch_size, 1024, 7, 1).to(device)  # [B, d_llm, N, 1]
    
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  x_mark: {x_mark.shape}")
    print(f"  emb: {emb.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(x, x_mark, emb)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, 96, 7]")
    
    # 验证输出形状
    assert output.shape == (batch_size, 96, 7), f"Output shape mismatch! Got {output.shape}, expected ({batch_size}, 96, 7)"
    
    # 检查是否有 NaN 或 Inf
    assert not torch.isnan(output).any(), "Output contains NaN!"
    assert not torch.isinf(output).any(), "Output contains Inf!"
    
    print("\n✅ V29 模型测试通过！")
    print("✅ 模型已成功回归 V18 结构（移除 Trend Projector，恢复 FFN=4*Channel）")
    
    # 检查关键组件
    print("\n关键组件检查:")
    print(f"  ✅ length_to_feature: {type(model.length_to_feature).__name__} (应该是 Linear)")
    print(f"  ✅ ts_encoder FFN: {model.ts_encoder[0].linear1.out_features} (应该是 {4*128})")
    print(f"  ✅ wp_encoder FFN: {model.wp_encoder.linear1.out_features} (应该是 {4*128})")
    print(f"  ✅ 无 trend_projector: {'trend_projector' not in dir(model)}")

if __name__ == "__main__":
    test_v29()
