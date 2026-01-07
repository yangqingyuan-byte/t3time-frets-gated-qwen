#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试 T3Time_FreTS_Gated_Qwen_LongSeq_v2 模型
"""
import torch
import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.T3Time_FreTS_Gated_Qwen_LongSeq_v2 import TriModalFreTSGatedQwenLongSeqV2

def test_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # 测试长序列预测长度
    for pred_len in [192, 336, 720]:
        print(f"\n测试 pred_len={pred_len}...")
        try:
            model = TriModalFreTSGatedQwenLongSeqV2(
                device=device, channel=32, num_nodes=7, seq_len=96, pred_len=pred_len,
                dropout_n=0.1, d_llm=1024, e_layer=1, d_layer=1, head=8,
                sparsity_threshold=0.009, frets_scale=0.018, use_dynamic_sparsity=True
            ).to(device)
            
            batch_size = 2
            input_data = torch.randn(batch_size, 96, 7).to(device)
            input_data_mark = torch.randn(batch_size, 96, 4).to(device)
            embeddings = torch.randn(batch_size, 1024, 7, 1).to(device)
            
            model.eval()
            with torch.no_grad():
                output = model(input_data, input_data_mark, embeddings)
            
            if output.shape == (batch_size, pred_len, 7) and not torch.isnan(output).any() and not torch.isinf(output).any():
                print(f"  ✅ pred_len={pred_len} 测试通过")
            else:
                print(f"  ❌ pred_len={pred_len} 测试失败")
                return False
        except Exception as e:
            print(f"  ❌ pred_len={pred_len} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n✅ 所有测试通过!")
    return True

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
