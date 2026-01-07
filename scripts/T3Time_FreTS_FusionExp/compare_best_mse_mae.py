#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比最佳MSE和最佳MAE参数组合的差异
"""
import json
import os
import sys
from collections import defaultdict

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_hyperopt_results(result_file=None, seed=2088, pred_len=720, model_id_prefix="T3Time_FreTS_Gated_Qwen_Hyperopt"):
    """加载参数寻优实验结果"""
    if result_file is None:
        result_file = os.path.join(project_root, "experiment_results.log")
    
    results = []
    
    if not os.path.exists(result_file):
        print(f"❌ 结果文件不存在: {result_file}")
        return results
    
    with open(result_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                # 检查是否是参数寻优实验结果
                if (data.get('seed') == seed and 
                    data.get('pred_len') == pred_len and
                    data.get('model_id', '').startswith(model_id_prefix)):
                    results.append(data)
            except json.JSONDecodeError as e:
                continue
            except Exception as e:
                continue
    
    return results

def compare_best_mse_mae(results):
    """对比最佳MSE和最佳MAE参数组合"""
    if not results:
        return None, None
    
    # 按 MSE 排序
    sorted_results_mse = sorted(results, key=lambda x: x.get('test_mse', float('inf')))
    best_mse = sorted_results_mse[0] if sorted_results_mse else None
    
    # 按 MAE 排序
    sorted_results_mae = sorted(results, key=lambda x: x.get('test_mae', float('inf')))
    best_mae = sorted_results_mae[0] if sorted_results_mae else None
    
    return best_mse, best_mae

def print_comparison(best_mse, best_mae, pred_len=720):
    """打印对比结果"""
    print("="*80)
    print(f"📊 预测长度 {pred_len} - 最佳MSE vs 最佳MAE 参数对比")
    print("="*80)
    
    if not best_mse or not best_mae:
        print("❌ 未找到实验结果")
        return
    
    # 定义所有要对比的参数
    params_to_compare = [
        ('data_path', '数据路径'),
        ('seq_len', '序列长度'),
        ('pred_len', '预测长度'),
        ('channel', 'Channel'),
        ('head', 'Head'),
        ('e_layer', 'E_Layer'),
        ('d_layer', 'D_Layer'),
        ('learning_rate', '学习率'),
        ('weight_decay', '权重衰减'),
        ('dropout_n', 'Dropout'),
        ('batch_size', '批次大小'),
        ('loss_fn', '损失函数'),
        ('lradj', '学习率调整'),
        ('embed_version', '嵌入版本'),
        ('epochs', '训练轮数'),
        ('patience', '早停耐心'),
        ('seed', '随机种子'),
    ]
    
    # 打印参数对比表
    print("\n" + "="*80)
    print("参数对比表")
    print("="*80)
    print(f"{'参数名称':<20} {'最佳MSE值':<25} {'最佳MAE值':<25} {'是否相同':<10}")
    print("-"*80)
    
    differences = []
    for param_key, param_name in params_to_compare:
        mse_value = best_mse.get(param_key, 'N/A')
        mae_value = best_mae.get(param_key, 'N/A')
        
        # 处理浮点数比较
        if isinstance(mse_value, float) and isinstance(mae_value, float):
            is_same = abs(mse_value - mae_value) < 1e-10
        else:
            is_same = mse_value == mae_value
        
        same_str = "✓ 相同" if is_same else "✗ 不同"
        
        # 格式化显示
        if isinstance(mse_value, float):
            mse_str = f"{mse_value:.6f}" if mse_value < 1 else f"{mse_value:.2e}"
        else:
            mse_str = str(mse_value)
        
        if isinstance(mae_value, float):
            mae_str = f"{mae_value:.6f}" if mae_value < 1 else f"{mae_value:.2e}"
        else:
            mae_str = str(mae_value)
        
        print(f"{param_name:<20} {mse_str:<25} {mae_str:<25} {same_str:<10}")
        
        if not is_same:
            differences.append((param_name, mse_value, mae_value))
    
    # 打印结果指标对比
    print("\n" + "="*80)
    print("结果指标对比")
    print("="*80)
    print(f"{'指标':<15} {'最佳MSE组合':<20} {'最佳MAE组合':<20} {'差异':<15}")
    print("-"*80)
    
    mse_mse = best_mse.get('test_mse', 0)
    mse_mae = best_mse.get('test_mae', 0)
    mae_mse = best_mae.get('test_mse', 0)
    mae_mae = best_mae.get('test_mae', 0)
    
    print(f"{'Test MSE':<15} {mse_mse:<20.6f} {mae_mse:<20.6f} {mae_mse - mse_mse:<15.6f}")
    print(f"{'Test MAE':<15} {mse_mae:<20.6f} {mae_mae:<20.6f} {mae_mae - mse_mae:<15.6f}")
    
    # 打印差异总结
    print("\n" + "="*80)
    print("差异总结")
    print("="*80)
    
    if not differences:
        print("✓ 两个参数组合完全相同！")
    else:
        print(f"发现 {len(differences)} 个参数不同：\n")
        for param_name, mse_value, mae_value in differences:
            print(f"  • {param_name}:")
            print(f"    - 最佳MSE组合: {mse_value}")
            print(f"    - 最佳MAE组合: {mae_value}")
            if isinstance(mse_value, float) and isinstance(mae_value, float):
                diff = abs(mae_value - mse_value)
                diff_pct = (diff / mse_value * 100) if mse_value != 0 else 0
                print(f"    - 差异: {diff:.6e} ({diff_pct:.2f}%)")
            print()
    
    # 解释为什么会有差异
    print("="*80)
    print("💡 为什么会有差异？")
    print("="*80)
    print("""
1. **优化目标不同**：
   - 最佳MSE组合：优化的是均方误差（MSE），对大误差更敏感
   - 最佳MAE组合：优化的是平均绝对误差（MAE），对所有误差同等对待

2. **参数影响**：
   - 学习率（learning_rate）和权重衰减（weight_decay）是正则化参数
   - 不同的优化目标可能需要不同的正则化强度
   - MSE更关注大误差，可能需要更强的正则化（更高的weight_decay）
   - MAE对所有误差同等对待，可能需要更温和的正则化

3. **实际意义**：
   - 如果更关注整体预测精度，使用最佳MSE组合
   - 如果更关注避免极端误差，使用最佳MAE组合
   - 两个组合的架构参数（channel, head, e_layer, d_layer）相同，说明模型结构是稳定的
    """)
    
    # 打印命令行格式对比
    print("\n" + "="*80)
    print("命令行格式对比（仅显示不同参数）")
    print("="*80)
    
    print("\n【最佳MSE组合】")
    print("python train_frets_gated_qwen.py \\")
    print(f"    --data_path {best_mse.get('data_path', 'ETTh1')} \\")
    print(f"    --seq_len {best_mse.get('seq_len', 96)} \\")
    print(f"    --pred_len {best_mse.get('pred_len', 720)} \\")
    print(f"    --channel {best_mse.get('channel', 'N/A')} \\")
    print(f"    --head {best_mse.get('head', 'N/A')} \\")
    print(f"    --e_layer {best_mse.get('e_layer', 1)} \\")
    print(f"    --d_layer {best_mse.get('d_layer', 1)} \\")
    print(f"    --learning_rate {best_mse.get('learning_rate', 'N/A')} \\  # ⚠️ 与MAE不同")
    print(f"    --weight_decay {best_mse.get('weight_decay', 'N/A')} \\  # ⚠️ 与MAE不同")
    print(f"    --dropout_n {best_mse.get('dropout_n', 'N/A')} \\")
    print(f"    --batch_size {best_mse.get('batch_size', 'N/A')} \\")
    print(f"    --loss_fn {best_mse.get('loss_fn', 'N/A')} \\")
    print(f"    --lradj {best_mse.get('lradj', 'type1')} \\")
    print(f"    --embed_version {best_mse.get('embed_version', 'qwen3_0.6b')} \\")
    print(f"    --epochs {best_mse.get('epochs', 100)} \\")
    print(f"    --es_patience {best_mse.get('patience', 10)} \\")
    print(f"    --seed {best_mse.get('seed', 2088)}")
    
    print("\n【最佳MAE组合】")
    print("python train_frets_gated_qwen.py \\")
    print(f"    --data_path {best_mae.get('data_path', 'ETTh1')} \\")
    print(f"    --seq_len {best_mae.get('seq_len', 96)} \\")
    print(f"    --pred_len {best_mae.get('pred_len', 720)} \\")
    print(f"    --channel {best_mae.get('channel', 'N/A')} \\")
    print(f"    --head {best_mae.get('head', 'N/A')} \\")
    print(f"    --e_layer {best_mae.get('e_layer', 1)} \\")
    print(f"    --d_layer {best_mae.get('d_layer', 1)} \\")
    print(f"    --learning_rate {best_mae.get('learning_rate', 'N/A')} \\  # ⚠️ 与MSE不同")
    print(f"    --weight_decay {best_mae.get('weight_decay', 'N/A')} \\  # ⚠️ 与MAE不同")
    print(f"    --dropout_n {best_mae.get('dropout_n', 'N/A')} \\")
    print(f"    --batch_size {best_mae.get('batch_size', 'N/A')} \\")
    print(f"    --loss_fn {best_mae.get('loss_fn', 'N/A')} \\")
    print(f"    --lradj {best_mae.get('lradj', 'type1')} \\")
    print(f"    --embed_version {best_mae.get('embed_version', 'qwen3_0.6b')} \\")
    print(f"    --epochs {best_mae.get('epochs', 100)} \\")
    print(f"    --es_patience {best_mae.get('patience', 10)} \\")
    print(f"    --seed {best_mae.get('seed', 2088)}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='对比最佳MSE和最佳MAE参数组合的差异')
    parser.add_argument('--result_file', type=str, default=None, help='结果文件路径（默认: experiment_results.log）')
    parser.add_argument('--seed', type=int, default=2088, help='随机种子')
    parser.add_argument('--pred_len', type=int, default=720, help='预测长度')
    parser.add_argument('--model_id_prefix', type=str, default='T3Time_FreTS_Gated_Qwen_Hyperopt', 
                       help='模型ID前缀')
    
    args = parser.parse_args()
    
    results = load_hyperopt_results(args.result_file, args.seed, args.pred_len, args.model_id_prefix)
    
    if not results:
        print(f"\n❌ 未找到 seed={args.seed}, pred_len={args.pred_len} 的参数寻优实验结果")
        return
    
    best_mse, best_mae = compare_best_mse_mae(results)
    print_comparison(best_mse, best_mae, args.pred_len)
    
    print("\n" + "="*80)
    print("对比完成！")
    print("="*80)

if __name__ == "__main__":
    main()
