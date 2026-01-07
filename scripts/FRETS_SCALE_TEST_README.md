# FreTS Component Scale 参数测试

## 📋 说明

测试不同的 `scale` 参数对 FreTS Component 初始化的影响，寻找最优值。

## 🎯 Scale 参数的作用

`scale` 参数控制 FreTS Component 中权重矩阵的初始化范围：
- `scale=0.01`: 较小的初始化范围，权重更接近零
- `scale=0.02`: 默认值，中等初始化范围
- `scale=0.05`: 较大的初始化范围，权重可能更大

**理论影响**:
- 较小的 scale：可能收敛更慢，但更稳定
- 较大的 scale：可能收敛更快，但可能不稳定

## 🚀 使用方法

### 方法 1: 批量测试（推荐）

```bash
bash scripts/test_frets_scale.sh
```

这会依次测试 scale=0.01, 0.02, 0.05，每个都使用相同的超参数：
- channel=128
- dropout=0.5
- weight_decay=1e-3
- loss_fn=mse
- lradj=type1
- sparsity_threshold=0.005

### 方法 2: 单个测试

```bash
# 测试 scale=0.01
python train_frets_gated_qwen_fusion_exp.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --channel 128 \
  --dropout_n 0.5 \
  --weight_decay 1e-3 \
  --fusion_mode gate \
  --loss_fn mse \
  --lradj type1 \
  --sparsity_threshold 0.005 \
  --frets_scale 0.01 \
  --seed 2024 \
  --epochs 100

# 测试 scale=0.02
python train_frets_gated_qwen_fusion_exp.py \
  --frets_scale 0.02 \
  ...

# 测试 scale=0.05
python train_frets_gated_qwen_fusion_exp.py \
  --frets_scale 0.05 \
  ...
```

## 📊 结果分析

训练完成后，所有结果会自动写入 `experiment_results.log`。

### 查看结果

```bash
# 查看所有 scale 测试的结果
grep "T3Time_FreTS_FusionExp_scale" experiment_results.log | \
  python -c "
import sys, json
results = []
for line in sys.stdin:
    data = json.loads(line.strip())
    results.append((data.get('frets_scale', 'unknown'), data['test_mse'], data['test_mae']))
results.sort(key=lambda x: x[1])  # 按 MSE 排序
print('Scale 参数对比 (按 MSE 排序):')
for scale, mse, mae in results:
    print(f'  scale={scale:4.2f} - MSE: {mse:.6f}, MAE: {mae:.6f}')
"
```

### 使用筛选脚本

```bash
python 筛选分析实验结果.py
# 然后选择: T3Time_FreTS_FusionExp_scale
```

## 💡 预期结果

根据经验：
- **scale=0.01**: 可能收敛较慢，但最终性能可能更好（更稳定）
- **scale=0.02**: 默认值，平衡性能和稳定性
- **scale=0.05**: 可能收敛较快，但可能不够稳定

**目标**: 找到能够达到 MSE < 0.383 的 scale 值

## 🔧 如果还不够

如果所有 scale 值都无法达到目标，可以考虑：

1. **调整 sparsity_threshold**
   - 当前: 0.005
   - 可以尝试: 0.001, 0.002, 0.01

2. **调整 horizon_info 权重**
   - 当前: `pred_len / 100.0`
   - 可以尝试: `pred_len / 50.0` 或 `pred_len / 200.0`

3. **尝试其他融合模式**
   - 当前: gate
   - 可以尝试: weighted, cross_attn, hybrid

## 📝 注意事项

1. 每个 scale 值的训练时间相同（100 epochs）
2. 使用相同的随机种子（2024）确保公平对比
3. 所有其他超参数保持一致
4. 结果会自动记录到实验日志
