#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试 T3Time_FreTS_Gated_Qwen_LongSeq_v2 模型
验证模型能否正常初始化和前向传播
"""
import torch
import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.T3Time_FreTS_Gated_Qwen_LongSeq_v2 import TriModalFreTSGatedQwenLongSeqV2

def test_model():
    """测试模型初始化和前向传播"""
    print("=" * 60)
    print("测试 T3Time_FreTS_Gated_Qwen_LongSeq_v2 模型")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 测试不同预测长度
    test_configs = [
        {"pred_len": 96, "name": "短序列"},
        {"pred_len": 192, "name": "中序列"},
        {"pred_len": 336, "name": "长序列"},
        {"pred_len": 720, "name": "超长序列"},
    ]
    
    for config in test_configs:
        pred_len = config["pred_len"]
        name = config["name"]
        print(f"\n{'='*60}")
        print(f"测试 {name} (pred_len={pred_len})")
        print(f"{'='*60}")
        
        try:
            # 创建模型
            model = TriModalFreTSGatedQwenLongSeqV2(
                device=device,
                channel=32,
                num_nodes=7,
                seq_len=96,
                pred_len=pred_len,
                dropout_n=0.1,
                d_llm=1024,
                e_layer=1,
                d_layer=1,
                head=8,
                sparsity_threshold=0.009,
                frets_scale=0.018,
                use_dynamic_sparsity=True
            ).to(device)
            
            print(f"✅ 模型创建成功")
            print(f"   可训练参数数量: {model.count_trainable_params():,}")
            
            # 创建测试数据
            batch_size = 2
            input_data = torch.randn(batch_size, 96, 7).to(device)
            input_data_mark = torch.randn(batch_size, 96, 4).to(device)
            embeddings = torch.randn(batch_size, 1024, 7, 1).to(device)
            
            print(f"   输入形状: input_data={input_data.shape}, embeddings={embeddings.shape}")
            
            # 前向传播
            model.eval()
            with torch.no_grad():
                output = model(input_data, input_data_mark, embeddings)
            
            expected_shape = (batch_size, pred_len, 7)
            if output.shape == expected_shape:
                print(f"✅ 前向传播成功")
                print(f"   输出形状: {output.shape} (期望: {expected_shape})")
            else:
                print(f"❌ 输出形状不匹配!")
                print(f"   实际: {output.shape}, 期望: {expected_shape}")
                return False
            
            # 检查输出是否包含 NaN 或 Inf
            if torch.isnan(output).any():
                print(f"❌ 输出包含 NaN!")
                return False
            if torch.isinf(output).any():
                print(f"❌ 输出包含 Inf!")
                return False
            
            print(f"✅ 输出数值正常 (范围: [{output.min().item():.4f}, {output.max().item():.4f}])")
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*60}")
    print("✅ 所有测试通过!")
    print(f"{'='*60}")
    return True

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
