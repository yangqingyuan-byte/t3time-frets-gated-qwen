"""
分析配置差异，找出为什么复现不出之前的结果
"""
import sys
import os
import json

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def analyze_config_differences():
    """分析配置差异"""
    log_file = "/root/0/T3Time/experiment_results.log"
    
    # 找出最佳结果
    best_result = None
    best_mse = float('inf')
    
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                model_id = data.get('model_id', '')
                
                if 'T3Time_FreTS_FusionExp' in model_id and data.get('pred_len') == 96:
                    mse = data.get('test_mse', float('inf'))
                    if mse < best_mse:
                        best_mse = mse
                        best_result = data
            except:
                continue
    
    if not best_result:
        print("未找到最佳结果")
        return
    
    print("="*80)
    print("最佳结果配置分析")
    print("="*80)
    
    print(f"\n[1] 最佳结果:")
    print(f"  MSE: {best_result.get('test_mse', 0):.6f}")
    print(f"  MAE: {best_result.get('test_mae', 0):.6f}")
    print(f"  Model ID: {best_result.get('model_id', 'unknown')}")
    print(f"  Timestamp: {best_result.get('timestamp', 'unknown')}")
    
    print(f"\n[2] 完整配置:")
    print(f"  Channel: {best_result.get('channel', 'N/A')}")
    print(f"  Dropout: {best_result.get('dropout_n', 'N/A')}")
    print(f"  Weight Decay: {best_result.get('weight_decay', 'N/A')}")
    print(f"  Scale: {best_result.get('frets_scale', 'N/A')} (N/A 表示使用默认值 0.02)")
    print(f"  Sparsity Threshold: {best_result.get('sparsity_threshold', 'N/A')} (N/A 表示使用默认值 0.01)")
    print(f"  Loss Function: {best_result.get('loss_fn', 'N/A')} (N/A 表示使用默认值 smooth_l1)")
    print(f"  Fusion Mode: {best_result.get('fusion_mode', 'N/A')}")
    print(f"  Seed: {best_result.get('seed', 'N/A')}")
    
    print(f"\n[3] 关键发现:")
    print("-" * 80)
    
    # 推断原始配置
    scale = best_result.get('frets_scale', None)
    sparsity = best_result.get('sparsity_threshold', None)
    loss_fn = best_result.get('loss_fn', None)
    
    inferred_config = {
        'scale': scale if scale is not None else 0.02,  # 原始默认值
        'sparsity_threshold': sparsity if sparsity is not None else 0.01,  # 原始默认值
        'loss_fn': loss_fn if loss_fn is not None else 'smooth_l1',  # 原始默认值
        'affine': True  # 原始默认值（在改动前）
    }
    
    print("推断的原始配置（基于日志中缺失的字段）:")
    print(f"  Scale: {inferred_config['scale']} (默认值，当时未指定)")
    print(f"  Sparsity Threshold: {inferred_config['sparsity_threshold']} (默认值，当时未指定)")
    print(f"  Loss Function: {inferred_config['loss_fn']} (默认值，当时未指定)")
    print(f"  Normalize affine: {inferred_config['affine']} (原始默认值，后来改成了 False)")
    
    print(f"\n[4] 当前代码的默认值:")
    print("-" * 80)
    print("  Scale: 0.02 (未改变)")
    print("  Sparsity Threshold: 0.005 (已改为 0.005)")
    print("  Loss Function: mse (已改为 mse)")
    print("  Normalize affine: False (已改为 False)")
    
    print(f"\n[5] 复现建议:")
    print("-" * 80)
    print("要复现最佳结果，需要使用原始配置：")
    print(f"  --channel {best_result.get('channel', 64)}")
    print(f"  --dropout_n {best_result.get('dropout_n', 0.1)}")
    print(f"  --weight_decay {best_result.get('weight_decay', 1e-4)}")
    print(f"  --frets_scale {inferred_config['scale']}")
    print(f"  --sparsity_threshold {inferred_config['sparsity_threshold']}")
    print(f"  --loss_fn {inferred_config['loss_fn']}")
    print(f"  --fusion_mode {best_result.get('fusion_mode', 'gate')}")
    print("")
    print("⚠️  注意: 还需要将 normalize_layers 改回 affine=True")
    print("   因为原始代码使用的是 affine=True")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_config_differences()
