"""
分析 FreTS FusionExp 参数寻优结果
找出最佳的 scale 和 sparsity_threshold 组合
"""
import sys
import os
import json

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def analyze_hyperopt_results():
    """分析参数寻优结果"""
    log_file = "/root/0/T3Time/experiment_results.log"
    results = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                model_id = data.get('model_id', '')
                
                # 筛选 HyperOpt 结果
                if 'T3Time_FreTS_FusionExp_HyperOpt' in model_id and data.get('pred_len') == 96:
                    results.append({
                        'scale': data.get('frets_scale', 0),
                        'sparsity': data.get('sparsity_threshold', 0),
                        'mse': data.get('test_mse', 0),
                        'mae': data.get('test_mae', 0),
                        'seed': data.get('seed', 'unknown')
                    })
            except:
                continue
    
    if not results:
        print("未找到 HyperOpt 实验结果")
        return
    
    print("="*80)
    print("FreTS FusionExp 参数寻优结果分析")
    print("="*80)
    
    # 按 MSE 排序
    results.sort(key=lambda x: x['mse'])
    
    print("\n[1] Top 10 最佳配置:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Scale':<8} {'Sparsity':<10} {'MSE':<12} {'MAE':<12} {'Seed':<8}")
    print("-" * 80)
    
    for i, r in enumerate(results[:10], 1):
        print(f"{i:<6} {r['scale']:<8.3f} {r['sparsity']:<10.3f} {r['mse']:<12.6f} {r['mae']:<12.6f} {r['seed']:<8}")
    
    # 最佳结果
    best = results[0]
    print(f"\n[2] 最佳配置:")
    print(f"  Scale: {best['scale']:.3f}")
    print(f"  Sparsity Threshold: {best['sparsity']:.3f}")
    print(f"  MSE: {best['mse']:.6f}")
    print(f"  MAE: {best['mae']:.6f}")
    print(f"  Seed: {best['seed']}")
    
    # 按配置分组统计
    print(f"\n[3] 配置统计（按 scale 和 sparsity 分组）:")
    print("-" * 80)
    
    config_groups = {}
    for r in results:
        key = (r['scale'], r['sparsity'])
        if key not in config_groups:
            config_groups[key] = []
        config_groups[key].append(r)
    
    # 计算每个配置的平均 MSE
    config_stats = []
    for (scale, sparsity), group in config_groups.items():
        avg_mse = sum(r['mse'] for r in group) / len(group)
        min_mse = min(r['mse'] for r in group)
        config_stats.append((scale, sparsity, avg_mse, min_mse, len(group)))
    
    config_stats.sort(key=lambda x: x[2])  # 按平均 MSE 排序
    
    print(f"{'Scale':<8} {'Sparsity':<10} {'平均 MSE':<12} {'最佳 MSE':<12} {'样本数':<8}")
    print("-" * 80)
    for scale, sparsity, avg_mse, min_mse, count in config_stats[:15]:
        print(f"{scale:<8.3f} {sparsity:<10.3f} {avg_mse:<12.6f} {min_mse:<12.6f} {count:<8}")
    
    # 与当前最佳对比
    print(f"\n[4] 与当前最佳对比:")
    print("-" * 80)
    current_best = 0.377142  # 当前最佳结果
    hyperopt_best = best['mse']
    diff = current_best - hyperopt_best
    
    if diff > 0:
        print(f"  ✅ HyperOpt 最佳结果优于当前最佳")
        print(f"     改进: {diff:.6f} ({abs(diff) / current_best * 100:.3f}%)")
    elif abs(diff) < 0.0001:
        print(f"  ✅ HyperOpt 最佳结果与当前最佳相当")
    else:
        print(f"  ⚠️  HyperOpt 最佳结果略差于当前最佳")
        print(f"     差距: {abs(diff):.6f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_hyperopt_results()
