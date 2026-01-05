"""
快速对比 T3Time_FreTS_FusionExp 与 T3Time_Pro_Qwen_SOTA_V30
重点分析性能差异的根本原因
"""
import sys
import os
import json

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_results():
    """从日志中加载结果"""
    log_file = "/root/0/T3Time/experiment_results.log"
    results = {}
    
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                model_id = data.get('model_id', '')
                pred_len = data.get('pred_len', 0)
                
                if pred_len == 96 and 'ETTh1' in data.get('data_path', ''):
                    if 'T3Time_Pro_Qwen_SOTA_V30' in model_id:
                        key = 'T3Time_V30'
                        if key not in results:
                            results[key] = []
                        results[key].append({
                            'mse': data.get('test_mse', 0),
                            'mae': data.get('test_mae', 0),
                            'seed': data.get('seed', 'unknown')
                        })
                    elif 'T3Time_FreTS_FusionExp' in model_id:
                        fusion_mode = data.get('fusion_mode', 'unknown')
                        key = f'FreTS_{fusion_mode}'
                        if key not in results:
                            results[key] = []
                        results[key].append({
                            'mse': data.get('test_mse', 0),
                            'mae': data.get('test_mae', 0),
                            'seed': data.get('seed', 'unknown')
                        })
            except:
                continue
    
    return results

def analyze_results():
    """分析结果"""
    results = load_results()
    
    print("="*80)
    print("实验结果对比分析")
    print("="*80)
    
    # 计算平均值
    print("\n[1] 平均性能对比 (pred_len=96):")
    print("-" * 80)
    
    t3time_results = results.get('T3Time_V30', [])
    if t3time_results:
        t3time_mse = sum(r['mse'] for r in t3time_results) / len(t3time_results)
        t3time_mae = sum(r['mae'] for r in t3time_results) / len(t3time_results)
        print(f"T3Time V30 (单模型):")
        print(f"  MSE: {t3time_mse:.6f} (范围: {min(r['mse'] for r in t3time_results):.6f} - {max(r['mse'] for r in t3time_results):.6f})")
        print(f"  MAE: {t3time_mae:.6f} (范围: {min(r['mae'] for r in t3time_results):.6f} - {max(r['mae'] for r in t3time_results):.6f})")
        print(f"  样本数: {len(t3time_results)}")
    
    print("\nFreTS FusionExp (各融合方式):")
    for key in sorted(results.keys()):
        if key.startswith('FreTS_'):
            fusion_mode = key.replace('FreTS_', '')
            frets_results = results[key]
            frets_mse = sum(r['mse'] for r in frets_results) / len(frets_results)
            frets_mae = sum(r['mae'] for r in frets_results) / len(frets_results)
            print(f"  {fusion_mode:12s}: MSE={frets_mse:.6f}, MAE={frets_mae:.6f} (样本数: {len(frets_results)})")
    
    # 关键差异分析
    print("\n" + "="*80)
    print("关键差异分析")
    print("="*80)
    
    print("\n[1] 架构差异:")
    print("  T3Time V30:")
    print("    - 频域: Learnable Wavelet Packet (4个频带)")
    print("    - 融合: Static Weights + Horizon-Aware Gate")
    print("    - Channel: 128, Dropout: 0.5, Weight Decay: 1e-3")
    print("  FreTS FusionExp:")
    print("    - 频域: FreTS Component (单频域表示)")
    print("    - 融合: Gate/Weighted/Cross-Attn/Hybrid")
    print("    - Channel: 64, Dropout: 0.1, Weight Decay: 1e-4")
    
    print("\n[2] 可能的原因:")
    reasons = [
        {
            "原因": "频域处理能力不足",
            "分析": "FreTS 使用单频域表示，而 T3Time 使用多频带分解。",
            "证据": "T3Time 的 4 个频带能够捕获不同频率成分，而 FreTS 可能丢失细节。",
            "优先级": "高"
        },
        {
            "原因": "正则化强度不足",
            "分析": "FreTS 使用 dropout=0.1, weight_decay=1e-4，而 T3Time 使用 dropout=0.5, weight_decay=1e-3。",
            "证据": "T3Time 的正则化强度是 FreTS 的 5 倍（dropout）和 10 倍（weight_decay）。",
            "优先级": "高"
        },
        {
            "原因": "模型容量不足",
            "分析": "FreTS 使用 channel=64，而 T3Time 使用 channel=128。",
            "证据": "参数量差异约 10%，可能限制了模型的学习能力。",
            "优先级": "中"
        },
        {
            "原因": "融合机制不够成熟",
            "分析": "实验性融合可能不如经过验证的静态权重机制稳定。",
            "证据": "T3Time 的静态权重机制经过了多轮优化，而实验性融合是新的尝试。",
            "优先级": "中"
        },
        {
            "原因": "缺少先验引导",
            "分析": "T3Time 使用先验引导初始化（低频优先），而 FreTS 使用随机初始化。",
            "证据": "先验引导可能帮助模型更快收敛到更好的解。",
            "优先级": "低"
        }
    ]
    
    for i, reason in enumerate(reasons, 1):
        print(f"\n  [{i}] {reason['优先级']}优先级 - {reason['原因']}")
        print(f"      分析: {reason['分析']}")
        print(f"      证据: {reason['证据']}")
    
    # 改进建议
    print("\n" + "="*80)
    print("改进建议")
    print("="*80)
    
    suggestions = [
        {
            "步骤": "1. 对齐超参数",
            "命令": "python train_frets_gated_qwen_fusion_exp.py --channel 128 --dropout_n 0.5 --weight_decay 1e-3 --fusion_mode gate",
            "预期": "验证是否是超参数导致的性能差异"
        },
        {
            "步骤": "2. 增强频域处理",
            "命令": "考虑引入多频带分解或增强 FreTS Component 容量",
            "预期": "提升频域特征提取能力"
        },
        {
            "步骤": "3. 引入静态权重",
            "命令": "参考 T3Time 的静态权重机制，添加先验引导初始化",
            "预期": "稳定融合机制"
        },
        {
            "步骤": "4. 调整训练策略",
            "命令": "使用 Step Decay 学习率调度器（T3Time V30 使用）",
            "预期": "改善训练稳定性"
        }
    ]
    
    for suggestion in suggestions:
        print(f"\n{suggestion['步骤']}")
        print(f"  方法: {suggestion['命令']}")
        print(f"  预期: {suggestion['预期']}")

if __name__ == "__main__":
    analyze_results()
