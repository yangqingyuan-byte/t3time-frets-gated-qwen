import json
import os
from collections import defaultdict

LOG_FILE = "experiment_results.log"

def analyze_results(model_filter=None):
    if not os.path.exists(LOG_FILE):
        print(f"错误: 找不到日志文件 {LOG_FILE}")
        return

    # 使用 defaultdict 按 data_path 分组
    grouped_results = defaultdict(list)
    
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # 如果指定了模型名，进行过滤（忽略大小写）
                if model_filter and model_filter.lower() not in data.get('model', '').lower():
                    continue
                
                dataset = data.get('data_path', 'Unknown')
                grouped_results[dataset].append(data)
            except json.JSONDecodeError:
                continue

    if not grouped_results:
        print("未找到任何实验结果。")
        return

    print("=" * 80)
    print(f"📊 实验结果分数据集深度分析报告")
    print(f"筛选模型关键词: {model_filter if model_filter else '全部'}")
    print("=" * 80)

    # 遍历每个数据集进行分析
    for dataset in sorted(grouped_results.keys()):
        results = grouped_results[dataset]
        print(f"\n📂 数据集: 【{dataset}】 (共 {len(results)} 条记录)")
        print("-" * 60)

        # 分别获取 MSE 和 MAE 的前 5 名
        top_5_mse = sorted(results, key=lambda x: x.get('test_mse', float('inf')))[:5]
        top_5_mae = sorted(results, key=lambda x: x.get('test_mae', float('inf')))[:5]

        # --- MSE 部分 ---
        print(f"🏆 [TOP 5 - 最小 TEST MSE]")
        for i, res in enumerate(top_5_mse):
            star = "⭐ " if i == 0 else "   "
            print(f" {star}第 {i+1} 名: MSE = {res['test_mse']:.6f} | MAE = {res['test_mae']:.6f}")
            print(f"     模型: {res.get('model', 'Unknown')}")
            
            # 动态拼接参数列表，使显示更整洁
            params = []
            if 'seed' in res: params.append(f"Seed={res['seed']}")
            if 'channel' in res: params.append(f"Channel={res['channel']}")
            if 'learning_rate' in res: params.append(f"LR={res['learning_rate']}")
            if 'dropout_n' in res: params.append(f"Dropout={res['dropout_n']}")
            if 'wavelet' in res: params.append(f"Wavelet={res['wavelet']}")
            
            print(f"     参数: {', '.join(params)}")
            print(f"     时间: {res.get('timestamp', 'N/A')}")

        # --- MAE 部分 ---
        print(f"\n🏆 [TOP 5 - 最小 TEST MAE]")
        for i, res in enumerate(top_5_mae):
            star = "⭐ " if i == 0 else "   "
            print(f" {star}第 {i+1} 名: MAE = {res['test_mae']:.6f} | MSE = {res['test_mse']:.6f}")
            print(f"     模型: {res.get('model', 'Unknown')}")
            
            params = []
            if 'seed' in res: params.append(f"Seed={res['seed']}")
            if 'channel' in res: params.append(f"Channel={res['channel']}")
            if 'learning_rate' in res: params.append(f"LR={res['learning_rate']}")
            if 'dropout_n' in res: params.append(f"Dropout={res['dropout_n']}")
            if 'wavelet' in res: params.append(f"Wavelet={res['wavelet']}")
            
            print(f"     参数: {', '.join(params)}")
            print(f"     时间: {res.get('timestamp', 'N/A')}")
        
        print("\n" + "." * 60)

    print("\n" + "=" * 80)
    print("💡 提示: 结果已按数据集独立分组。若 MSE 和 MAE 榜首不同，请根据具体研究侧重点选择。")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    # 支持命令行参数指定过滤关键字，如: python scripts/find_best_results.py Refine
    filter_keyword = sys.argv[1] if len(sys.argv) > 1 else None
    analyze_results(filter_keyword)
