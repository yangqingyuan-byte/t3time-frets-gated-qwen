#!/bin/bash
# 快速查看 scale 参数测试结果

set -euo pipefail

echo "=========================================="
echo "FreTS Component Scale 参数结果查看"
echo "=========================================="
echo ""

# 使用分析脚本
python scripts/analyze_scale_results.py

echo ""
echo "=========================================="
echo "详细数据（按 MSE 排序）"
echo "=========================================="
echo ""

python3 << 'PYTHON_SCRIPT'
import sys
import json

# 读取日志文件
log_file = "/root/0/T3Time/experiment_results.log"
results = []

try:
    with open(log_file, 'r') as f:
        for line in f:
            if 'T3Time_FreTS_FusionExp_scale' in line:
                try:
                    data = json.loads(line.strip())
                    scale = data.get('frets_scale', 'unknown')
                    mse = data.get('test_mse', 0)
                    mae = data.get('test_mae', 0)
                    seed = data.get('seed', 'unknown')
                    results.append((float(scale), mse, mae, seed))
                except Exception as e:
                    print(f'Error parsing line: {e}', file=sys.stderr)
                    continue
except FileNotFoundError:
    print(f"日志文件未找到: {log_file}")
    sys.exit(1)

if not results:
    print("未找到相关实验结果")
    sys.exit(1)

# 按 MSE 排序
results.sort(key=lambda x: x[1])

print(f"{'Scale':<10} {'MSE':<15} {'MAE':<15} {'Seed':<10}")
print("-" * 50)
for scale, mse, mae, seed in results:
    print(f"{scale:<10.3f} {mse:<15.6f} {mae:<15.6f} {seed:<10}")

# 找出最佳
if results:
    best = results[0]
    print("")
    print(f"✅ 最佳结果: scale={best[0]:.3f}, MSE={best[1]:.6f}, MAE={best[2]:.6f}")
    print(f"   与 T3Time V30 (0.3835) 对比: 优势 {0.3835 - best[1]:.6f}")
PYTHON_SCRIPT
