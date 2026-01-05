"""
分析 FreTS Component scale 参数对性能的影响
"""
import sys
import os
import json

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def analyze_scale_results():
    """分析 scale 参数的结果"""
    log_file = "/root/0/T3Time/experiment_results.log"
    results = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                model_id = data.get('model_id', '')
                if 'T3Time_FreTS_FusionExp_scale' in model_id and data.get('pred_len') == 96:
                    scale = data.get('frets_scale', 'unknown')
                    mse = data.get('test_mse', 0)
                    mae = data.get('test_mae', 0)
                    seed = data.get('seed', 'unknown')
                    results.append({
                        'scale': scale,
                        'mse': mse,
                        'mae': mae,
                        'seed': seed,
                        'model_id': model_id
                    })
            except:
                continue
    
    if not results:
        print("未找到相关实验结果")
        return
    
    # 按 scale 分组
    scale_groups = {}
    for r in results:
        scale = r['scale']
        if scale not in scale_groups:
            scale_groups[scale] = []
        scale_groups[scale].append(r)
    
    print("="*80)
    print("FreTS Component Scale 参数性能分析")
    print("="*80)
    
    # 按 scale 排序
    sorted_scales = sorted(scale_groups.keys())
    
    print("\n[1] 各 Scale 值的性能:")
    print("-" * 80)
    print(f"{'Scale':<10} {'MSE':<15} {'MAE':<15} {'样本数':<10}")
    print("-" * 80)
    
    for scale in sorted_scales:
        group = scale_groups[scale]
        avg_mse = sum(r['mse'] for r in group) / len(group)
        avg_mae = sum(r['mae'] for r in group) / len(group)
        min_mse = min(r['mse'] for r in group)
        max_mse = max(r['mse'] for r in group)
        print(f"{scale:<10.3f} {avg_mse:<15.6f} {avg_mae:<15.6f} {len(group):<10}")
        if len(group) > 1:
            print(f"           (范围: {min_mse:.6f} - {max_mse:.6f})")
    
    # 找出最佳 scale
    best_result = min(results, key=lambda x: x['mse'])
    print(f"\n[2] 最佳 Scale 值:")
    print(f"  Scale: {best_result['scale']}")
    print(f"  MSE: {best_result['mse']:.6f}")
    print(f"  MAE: {best_result['mae']:.6f}")
    print(f"  Seed: {best_result['seed']}")
    print(f"  Model ID: {best_result['model_id']}")
    
    # 分析趋势
    print(f"\n[3] 性能趋势分析:")
    if len(sorted_scales) >= 2:
        first_mse = sum(r['mse'] for r in scale_groups[sorted_scales[0]]) / len(scale_groups[sorted_scales[0]])
        last_mse = sum(r['mse'] for r in scale_groups[sorted_scales[-1]]) / len(scale_groups[sorted_scales[-1]])
        
        if first_mse < last_mse:
            print(f"  ✅ 趋势: 随着 scale 增加，MSE 增加（性能下降）")
            print(f"     建议: 尝试更小的 scale 值（< {sorted_scales[0]}）")
        elif first_mse > last_mse:
            print(f"  ✅ 趋势: 随着 scale 增加，MSE 减少（性能提升）")
            print(f"     建议: 尝试更大的 scale 值（> {sorted_scales[-1]}）")
        else:
            print(f"  ⚠️  趋势: scale 对性能影响不明显")
    
    # 与 T3Time V30 对比
    print(f"\n[4] 与 T3Time V30 对比:")
    t3time_mse = 0.3835  # T3Time V30 集成结果
    best_mse = best_result['mse']
    diff = best_mse - t3time_mse
    if diff < 0:
        print(f"  ✅ 最佳结果 ({best_result['scale']}) 优于 T3Time V30")
        print(f"     优势: {abs(diff):.6f}")
    elif diff < 0.001:
        print(f"  ✅ 最佳结果 ({best_result['scale']}) 接近 T3Time V30")
        print(f"     差异: {diff:.6f} (可忽略)")
    else:
        print(f"  ⚠️  最佳结果 ({best_result['scale']}) 仍不如 T3Time V30")
        print(f"     差距: {diff:.6f}")
        print(f"     建议: 继续尝试更小的 scale 值或其他参数调整")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_scale_results()
