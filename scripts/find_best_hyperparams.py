import json
import os
import collections
import statistics
import argparse
from tabulate import tabulate

def load_logs(log_file):
    data = []
    if not os.path.exists(log_file):
        return []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except:
                continue
    return data

def get_param_key(log):
    # 忽略 seed 和 timestamp，保留核心超参数
    exclude = {'seed', 'timestamp', 'test_mse', 'test_mae', 'data_path', 'pred_len'}
    params = {k: v for k, v in log.items() if k not in exclude}
    # 排序键以确保一致性
    sorted_params = sorted(params.items())
    return tuple(sorted_params)

def analyze(log_file, data_path=None, pred_len=None, top_n=10):
    logs = load_logs(log_file)
    if not logs:
        print("未找到日志数据。")
        return

    # 预筛选
    if data_path:
        logs = [l for l in logs if l.get('data_path') == data_path]
    if pred_len:
        logs = [l for l in logs if l.get('pred_len') == pred_len]

    if not logs:
        print(f"在筛选条件 (data_path={data_path}, pred_len={pred_len}) 下没有数据。")
        return

    # 按超参数组合分组
    groups = collections.defaultdict(list)
    for l in logs:
        key = get_param_key(l)
        groups[key].append(l)

    results = []
    for key, group_logs in groups.items():
        mses = [l['test_mse'] for l in group_logs]
        maes = [l['test_mae'] for l in group_logs]
        
        avg_mse = statistics.mean(mses)
        std_mse = statistics.stdev(mses) if len(mses) > 1 else 0
        avg_mae = statistics.mean(maes)
        std_mae = statistics.stdev(maes) if len(maes) > 1 else 0
        
        # 还原参数字典用于展示
        params = dict(key)
        results.append({
            'params': params,
            'avg_mse': avg_mse,
            'std_mse': std_mse,
            'avg_mae': avg_mae,
            'std_mae': std_mae,
            'count': len(group_logs),
            'min_mse': min(mses),
            'min_mae': min(maes)
        })

    # 按平均 MSE 排序
    results.sort(key=lambda x: x['avg_mse'])

    print(f"\n" + "="*100)
    print(f" 超参数组合分析报告 (Data={data_path or 'All'}, Pred_Len={pred_len or 'All'})")
    print(f" 目标：寻找最稳健的高性能参数 (按平均 MSE 排序)")
    print("="*100)

    table = []
    for i, res in enumerate(results[:top_n]):
        param_str = "\n".join([f"{k}: {v}" for k, v in res['params'].items()])
        table.append([
            i + 1,
            f"{res['avg_mse']:.6f} ± {res['std_mse']:.4f}\n(Min: {res['min_mse']:.6f})",
            f"{res['avg_mae']:.6f} ± {res['std_mae']:.4f}\n(Min: {res['min_mae']:.6f})",
            res['count'],
            param_str
        ])

    print(tabulate(table, headers=["Rank", "Avg MSE ± Std", "Avg MAE ± Std", "Runs", "Hyperparameters"], tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='/root/0/T3Time/experiment_results.log')
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()

    analyze(args.log, args.data, args.pred_len, args.top)

