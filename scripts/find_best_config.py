"""
找出最佳配置，对比所有相关实验
"""
import sys
import os
import json

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def find_best_config():
    """找出最佳配置"""
    log_file = "/root/0/T3Time/experiment_results.log"
    results = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                model_id = data.get('model_id', '')
                
                # 筛选 FreTS FusionExp 的结果
                if 'T3Time_FreTS_FusionExp' in model_id and data.get('pred_len') == 96:
                    results.append({
                        'model_id': model_id,
                        'mse': data.get('test_mse', 0),
                        'mae': data.get('test_mae', 0),
                        'channel': data.get('channel', 0),
                        'dropout': data.get('dropout_n', 0),
                        'weight_decay': data.get('weight_decay', 0),
                        'scale': data.get('frets_scale', 'N/A'),
                        'sparsity_threshold': data.get('sparsity_threshold', 'N/A'),
                        'loss_fn': data.get('loss_fn', 'N/A'),
                        'fusion_mode': data.get('fusion_mode', 'N/A'),
                        'seed': data.get('seed', 'unknown'),
                        'timestamp': data.get('timestamp', 'unknown')
                    })
            except:
                continue
    
    if not results:
        print("未找到相关实验结果")
        return
    
    # 按 MSE 排序
    results.sort(key=lambda x: x['mse'])
    
    print("="*80)
    print("FreTS FusionExp 最佳配置分析")
    print("="*80)
    
    print("\n[1] Top 5 最佳结果:")
    print("-" * 80)
    print(f"{'Rank':<6} {'MSE':<12} {'MAE':<12} {'Channel':<8} {'Dropout':<8} {'WD':<10} {'Scale':<8} {'Sparsity':<10}")
    print("-" * 80)
    
    for i, r in enumerate(results[:5], 1):
        print(f"{i:<6} {r['mse']:<12.6f} {r['mae']:<12.6f} {r['channel']:<8} {r['dropout']:<8.1f} {r['weight_decay']:<10.0e} {str(r['scale']):<8} {str(r['sparsity_threshold']):<10}")
    
    # 最佳结果详情
    best = results[0]
    print(f"\n[2] 最佳配置详情:")
    print("-" * 80)
    print(f"  Model ID: {best['model_id']}")
    print(f"  MSE: {best['mse']:.6f}")
    print(f"  MAE: {best['mae']:.6f}")
    print(f"  Channel: {best['channel']}")
    print(f"  Dropout: {best['dropout']}")
    print(f"  Weight Decay: {best['weight_decay']}")
    print(f"  Scale: {best['scale']}")
    print(f"  Sparsity Threshold: {best['sparsity_threshold']}")
    print(f"  Loss Function: {best['loss_fn']}")
    print(f"  Fusion Mode: {best['fusion_mode']}")
    print(f"  Seed: {best['seed']}")
    print(f"  Timestamp: {best['timestamp']}")
    
    # 分析配置模式
    print(f"\n[3] 配置模式分析:")
    print("-" * 80)
    
    # 按配置分组
    config_groups = {}
    for r in results:
        key = f"c{r['channel']}_d{r['dropout']}_wd{r['weight_decay']}"
        if key not in config_groups:
            config_groups[key] = []
        config_groups[key].append(r)
    
    print("各配置的平均性能:")
    for key in sorted(config_groups.keys()):
        group = config_groups[key]
        avg_mse = sum(r['mse'] for r in group) / len(group)
        best_mse = min(r['mse'] for r in group)
        print(f"  {key}: 平均 MSE={avg_mse:.6f}, 最佳 MSE={best_mse:.6f} (样本数: {len(group)})")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    find_best_config()
