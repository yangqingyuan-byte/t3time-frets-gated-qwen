"""
对比不同超参数配置的性能差异
分析为什么对齐 T3Time V30 超参数后性能反而下降
"""
import sys
import os
import json

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def compare_hyperparams():
    """对比不同超参数配置"""
    log_file = "/root/0/T3Time/experiment_results.log"
    
    # 配置1: 原始配置（channel=64, dropout=0.1, weight_decay=1e-4）
    config1_results = []
    # 配置2: 对齐 T3Time V30（channel=128, dropout=0.5, weight_decay=1e-3）
    config2_results = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                model_id = data.get('model_id', '')
                
                # 筛选 FreTS FusionExp 的结果
                if 'T3Time_FreTS_FusionExp' in model_id and data.get('pred_len') == 96:
                    channel = data.get('channel', 0)
                    dropout = data.get('dropout_n', 0)
                    weight_decay = data.get('weight_decay', 0)
                    mse = data.get('test_mse', 0)
                    mae = data.get('test_mae', 0)
                    scale = data.get('frets_scale', 'N/A')
                    
                    result = {
                        'model_id': model_id,
                        'channel': channel,
                        'dropout': dropout,
                        'weight_decay': weight_decay,
                        'scale': scale,
                        'mse': mse,
                        'mae': mae,
                        'seed': data.get('seed', 'unknown')
                    }
                    
                    # 配置1: channel=64, dropout=0.1, weight_decay=1e-4
                    if channel == 64 and abs(dropout - 0.1) < 0.01 and abs(weight_decay - 1e-4) < 1e-6:
                        config1_results.append(result)
                    # 配置2: channel=128, dropout=0.5, weight_decay=1e-3
                    elif channel == 128 and abs(dropout - 0.5) < 0.01 and abs(weight_decay - 1e-3) < 1e-6:
                        config2_results.append(result)
            except:
                continue
    
    print("="*80)
    print("超参数配置对比分析")
    print("="*80)
    
    print("\n[1] 配置1: 原始配置（channel=64, dropout=0.1, weight_decay=1e-4）")
    print("-" * 80)
    if config1_results:
        for r in config1_results:
            print(f"  Model: {r['model_id']}")
            print(f"    MSE: {r['mse']:.6f}, MAE: {r['mae']:.6f}")
            print(f"    Scale: {r['scale']}, Seed: {r['seed']}")
        avg_mse1 = sum(r['mse'] for r in config1_results) / len(config1_results)
        avg_mae1 = sum(r['mae'] for r in config1_results) / len(config1_results)
        print(f"\n  平均: MSE={avg_mse1:.6f}, MAE={avg_mae1:.6f}")
        best1 = min(config1_results, key=lambda x: x['mse'])
        print(f"  最佳: MSE={best1['mse']:.6f}, MAE={best1['mae']:.6f}")
    else:
        print("  未找到相关结果")
    
    print("\n[2] 配置2: 对齐 T3Time V30（channel=128, dropout=0.5, weight_decay=1e-3）")
    print("-" * 80)
    if config2_results:
        # 按 scale 分组
        scale_groups = {}
        for r in config2_results:
            scale = r['scale']
            if scale not in scale_groups:
                scale_groups[scale] = []
            scale_groups[scale].append(r)
        
        print(f"  测试了 {len(scale_groups)} 个不同的 scale 值:")
        for scale in sorted(scale_groups.keys()):
            group = scale_groups[scale]
            avg_mse = sum(r['mse'] for r in group) / len(group)
            print(f"    scale={scale:5.3f}: MSE={avg_mse:.6f} (样本数: {len(group)})")
        
        avg_mse2 = sum(r['mse'] for r in config2_results) / len(config2_results)
        avg_mae2 = sum(r['mae'] for r in config2_results) / len(config2_results)
        print(f"\n  平均: MSE={avg_mse2:.6f}, MAE={avg_mae2:.6f}")
        best2 = min(config2_results, key=lambda x: x['mse'])
        print(f"  最佳: MSE={best2['mse']:.6f}, MAE={best2['mae']:.6f}, scale={best2['scale']}")
    else:
        print("  未找到相关结果")
    
    # 对比分析
    print("\n[3] 对比分析")
    print("-" * 80)
    if config1_results and config2_results:
        best1 = min(config1_results, key=lambda x: x['mse'])
        best2 = min(config2_results, key=lambda x: x['mse'])
        
        print(f"配置1 最佳: MSE={best1['mse']:.6f}")
        print(f"配置2 最佳: MSE={best2['mse']:.6f}")
        diff = best2['mse'] - best1['mse']
        
        if diff > 0:
            print(f"\n⚠️  配置2 比配置1 差 {diff:.6f}")
            print(f"   配置1 更优！")
        else:
            print(f"\n✅ 配置2 比配置1 好 {abs(diff):.6f}")
    
    # 原因分析
    print("\n[4] 可能的原因分析")
    print("-" * 80)
    reasons = [
        {
            "原因": "正则化过强导致欠拟合",
            "分析": "dropout=0.5 和 weight_decay=1e-3 对于 FreTS 模型可能过强，导致模型学习能力受限",
            "证据": "配置1 使用较小的正则化（dropout=0.1, weight_decay=1e-4）性能更好"
        },
        {
            "原因": "模型容量与数据不匹配",
            "分析": "channel=128 虽然增加了模型容量，但配合强正则化可能导致欠拟合",
            "证据": "配置1 使用 channel=64 配合轻正则化，性能更好"
        },
        {
            "原因": "FreTS 与 T3Time 的架构差异",
            "分析": "FreTS 使用单频域表示，而 T3Time 使用多频带分解。不同的架构可能需要不同的超参数",
            "证据": "T3Time V30 的超参数可能不适合 FreTS 架构"
        },
        {
            "原因": "训练策略差异",
            "分析": "虽然都使用 Step Decay，但不同的正则化强度可能影响学习率的效果",
            "证据": "需要进一步分析训练曲线"
        }
    ]
    
    for i, reason in enumerate(reasons, 1):
        print(f"\n  [{i}] {reason['原因']}")
        print(f"      分析: {reason['分析']}")
        print(f"      证据: {reason['证据']}")
    
    # 建议
    print("\n[5] 改进建议")
    print("-" * 80)
    suggestions = [
        "1. 使用配置1 的超参数（channel=64, dropout=0.1, weight_decay=1e-4）作为基础",
        "2. 在此基础上优化 scale 参数，寻找最佳值",
        "3. 或者尝试中间配置：channel=96, dropout=0.3, weight_decay=5e-4",
        "4. 分析训练曲线，看是否存在过拟合或欠拟合",
        "5. 考虑 FreTS 架构的特殊性，可能需要专门调优"
    ]
    
    for suggestion in suggestions:
        print(f"  {suggestion}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    compare_hyperparams()
