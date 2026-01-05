#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试 T3Time_FreEformer_Gated_Qwen 模型
"""
import torch
import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.T3Time_FreEformer_Gated_Qwen import TriModalFreEformerGatedQwen

def test_model():
    """测试模型的基本功能"""
    print("="*80)
    print("测试 T3Time_FreEformer_Gated_Qwen 模型")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")
    
    # 模型参数
    num_nodes = 7
    seq_len = 96
    pred_lens = [96, 192, 336, 720]
    batch_size = 2
    d_llm = 1024
    
    # 测试不同预测长度
    for pred_len in pred_lens:
        print(f"\n{'='*80}")
        print(f"测试 pred_len={pred_len}")
        print(f"{'='*80}")
        
        try:
            # 创建模型
            model = TriModalFreEformerGatedQwen(
                device=device,
                channel=64,
                num_nodes=num_nodes,
                seq_len=seq_len,
                pred_len=pred_len,
                dropout_n=0.1,
                d_llm=d_llm,
                e_layer=1,
                d_layer=1,
                d_ff=32,
                head=8,
                embed_size=16,
                fre_e_layer=1
            ).to(device)
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = model.count_trainable_params()
            print(f"总参数量: {total_params:,}")
            print(f"可训练参数量: {trainable_params:,}")
            
            # 创建输入
            input_data = torch.randn(batch_size, seq_len, num_nodes).to(device)
            input_data_mark = None  # T3Time_FreTS_Gated_Qwen 不使用这个
            embeddings = torch.randn(batch_size, d_llm, num_nodes, 1).to(device)
            
            print(f"\n输入形状:")
            print(f"  input_data: {input_data.shape}")
            print(f"  embeddings: {embeddings.shape}")
            
            # 前向传播
            model.eval()
            with torch.no_grad():
                output = model(input_data, input_data_mark, embeddings)
            
            print(f"\n输出形状: {output.shape}")
            print(f"期望形状: ({batch_size}, {pred_len}, {num_nodes})")
            
            # 检查输出形状
            assert output.shape == (batch_size, pred_len, num_nodes), \
                f"输出形状不匹配！期望 ({batch_size}, {pred_len}, {num_nodes}), 得到 {output.shape}"
            
            print(f"✅ pred_len={pred_len} 测试通过！")
            
            # 检查数值稳定性
            if torch.isnan(output).any():
                print(f"⚠️  警告: 输出包含 NaN")
            if torch.isinf(output).any():
                print(f"⚠️  警告: 输出包含 Inf")
            
            print(f"输出统计: min={output.min().item():.6f}, max={output.max().item():.6f}, mean={output.mean().item():.6f}")
            
        except Exception as e:
            print(f"❌ pred_len={pred_len} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*80}")
    print("✅ 所有测试通过！")
    print(f"{'='*80}")
    return True

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
