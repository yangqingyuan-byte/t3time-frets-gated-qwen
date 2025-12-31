import json
import os
import statistics
from collections import defaultdict
from tabulate import tabulate

def analyze_ablation(log_file="/root/0/T3Time/ablation_results.log"):
    if not os.path.exists(log_file):
        print(f"错误: 找不到结果文件 {log_file}")
        return

    data = defaultdict(list)
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith("开始消歧"):
                continue
            try:
                # 兼容脚本输出的 json
                entry = json.loads(line.strip())
                data[entry['experiment']].append(entry)
            except:
                continue

    if not data:
        print("日志文件中尚未包含有效数据。")
        return

    summary = []
    for exp_name, results in data.items():
        mses = [float(r['mse']) for r in results if r['mse']]
        maes = [float(r['mae']) for r in results if r['mae']]
        
        if not mses: continue
        
        avg_mse = statistics.mean(mses)
        std_mse = statistics.stdev(mses) if len(mses) > 1 else 0
        avg_mae = statistics.mean(maes)
        
        summary.append({
            "Name": exp_name,
            "Avg MSE": avg_mse,
            "Std": std_mse,
            "Avg MAE": avg_mae,
            "Runs": len(results)
        })

    # 按 MSE 排序
    summary.sort(key=lambda x: x['Avg MSE'])

    print("\n" + "="*80)
    print(" 消歧实验自动分析报告")
    print("="*80)
    
    table = []
    for i, s in enumerate(summary):
        table.append([
            i+1, 
            s['Name'], 
            f"{s['Avg MSE']:.6f}", 
            f"±{s['Std']:.4f}", 
            f"{s['Avg MAE']:.6f}", 
            s['Runs']
        ])
    
    print(tabulate(table, headers=["Rank", "Experiment Name", "Avg MSE", "Stability", "Avg MAE", "Seeds"], tablefmt="grid"))

    # --- 核心洞察逻辑 ---
    print("\n[核心结论分析]")
    
    def get_mse(name):
        for s in summary:
            if s['Name'] == name: return s['Avg MSE']
        return None

    frets_mse = get_mse("FreTS_Arch_MSE_Loss")
    fredf_mse = get_mse("FFT_Arch_FreDF_Loss")
    base_mse = get_mse("FFT_Arch_MSE_Loss")
    both_mse = get_mse("FreTS_Arch_FreDF_Loss")

    if frets_mse and base_mse:
        diff = ((frets_mse - base_mse) / base_mse) * 100
        status = "提升" if diff < 0 else "下降"
        print(f"1. 架构对比: FreTS 相比 FFT 性能{status}了 {abs(diff):.2f}%")

    if fredf_mse and base_mse:
        diff = ((fredf_mse - base_mse) / base_mse) * 100
        status = "有效" if diff < 0 else "起反作用"
        print(f"2. 损失函数: FreDF 相比 MSE {status}，影响幅度为 {abs(diff):.2f}%")

    if both_mse and frets_mse:
        diff = ((both_mse - frets_mse) / frets_mse) * 100
        if diff > 0:
            print(f"3. 冲突警告: 同时使用 FreTS 和 FreDF 导致性能进一步下降了 {abs(diff):.2f}%，可能存在频域信息过拟合。")
        else:
            print(f"3. 协同效应: 同时使用 FreTS 和 FreDF 产生了正面增益！")

if __name__ == "__main__":
    analyze_ablation()

