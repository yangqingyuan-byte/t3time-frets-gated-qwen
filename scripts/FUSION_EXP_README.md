# T3Time_FreTS_Gated_Qwen 融合机制实验

## 📋 实验概述

本实验对比 4 种不同的时域-频域融合机制，找出最适合 FreTS 模型的融合方式。

## 🔬 实验版本

### 版本 A: Gate 融合（Horizon-Aware Gate）
- **机制**: 类似 T3Time V30 的门控融合
- **公式**: `fused = time + gate(time, freq, horizon) * freq`
- **特点**: 引入预测长度信息，自适应控制频域信息融入程度
- **参数**: `--fusion_mode gate`

### 版本 B: Weighted 融合（可学习加权求和）
- **机制**: 简单的可学习加权求和
- **公式**: `fused = α * time + (1-α) * freq`，其中 α 是可学习参数
- **特点**: 最简单，参数量最少
- **参数**: `--fusion_mode weighted`

### 版本 C: Cross-Attn 融合（改进的 Cross-Attention）
- **机制**: Cross-Attention + 双重残差连接
- **公式**: `fused = Norm(Attn(time, freq, freq) + time + freq)`
- **特点**: 保留原始 Cross-Attention，但增加频域残差连接
- **参数**: `--fusion_mode cross_attn`

### 版本 D: Hybrid 融合（混合融合）
- **机制**: Cross-Attention + 门控
- **公式**: `fused = Norm(Attn + time) + gate(time, freq) * freq`
- **特点**: 结合 Cross-Attention 和门控的优势
- **参数**: `--fusion_mode hybrid`

## 🚀 使用方法

### 方法 1: 单个实验

```bash
# 测试 Gate 融合
python train_frets_gated_qwen_fusion_exp.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --channel 64 \
  --dropout_n 0.1 \
  --weight_decay 1e-4 \
  --fusion_mode gate \
  --seed 2024

# 测试 Weighted 融合
python train_frets_gated_qwen_fusion_exp.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --channel 64 \
  --dropout_n 0.1 \
  --weight_decay 1e-4 \
  --fusion_mode weighted \
  --seed 2024

# 测试 Cross-Attn 融合
python train_frets_gated_qwen_fusion_exp.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --channel 64 \
  --dropout_n 0.1 \
  --weight_decay 1e-4 \
  --fusion_mode cross_attn \
  --seed 2024

# 测试 Hybrid 融合
python train_frets_gated_qwen_fusion_exp.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --channel 64 \
  --dropout_n 0.1 \
  --weight_decay 1e-4 \
  --fusion_mode hybrid \
  --seed 2024
```

### 方法 2: 批量对比实验

```bash
# 运行所有融合方式的对比实验
bash scripts/run_fusion_exp.sh
```

这会依次运行 4 种融合方式，每个都使用相同的超参数，便于公平对比。

## 📊 结果分析

训练完成后，所有结果会自动写入 `experiment_results.log`，格式如下：

```json
{
  "model_id": "T3Time_FreTS_FusionExp_gate",
  "fusion_mode": "gate",
  "test_mse": 0.xxx,
  "test_mae": 0.xxx,
  ...
}
```

### 查看结果

```bash
# 使用筛选脚本查看结果
python 筛选分析实验结果.py
# 选择: T3Time_FreTS_Gated_Qwen_FusionExp
```

### 快速对比

```bash
# 查看所有融合实验的结果
grep "T3Time_FreTS_FusionExp" experiment_results.log | \
  python -c "
import sys, json
results = []
for line in sys.stdin:
    data = json.loads(line.strip())
    results.append((data['fusion_mode'], data['test_mse'], data['test_mae']))
results.sort(key=lambda x: x[1])  # 按 MSE 排序
print('融合方式对比 (按 MSE 排序):')
for mode, mse, mae in results:
    print(f'  {mode:12s} - MSE: {mse:.6f}, MAE: {mae:.6f}')
"
```

## 🎯 预期发现

根据诊断结果，我们预期：

1. **Gate 融合** 可能表现最好（参考 T3Time V30 的成功经验）
2. **Cross-Attn 融合** 可能改善原始 Cross-Attention 的问题
3. **Weighted 融合** 最简单，但可能不够灵活
4. **Hybrid 融合** 可能过复杂，导致过拟合

## 🔧 超参数建议

如果某个融合方式表现好，可以进一步调优：

```bash
# 尝试 T3Time V30 的超参数
python train_frets_gated_qwen_fusion_exp.py \
  --fusion_mode gate \
  --channel 128 \
  --dropout_n 0.5 \
  --weight_decay 1e-3 \
  --learning_rate 5e-5
```

## 📝 实验记录

建议记录：
- 每种融合方式的训练曲线
- 最终测试 MSE/MAE
- 训练时间
- 模型参数量差异

## 🐛 故障排除

如果遇到问题：

1. **导入错误**: 确保在项目根目录运行
2. **CUDA 错误**: 检查 GPU 可用性和内存
3. **数据加载错误**: 确认数据集路径正确

## 📚 相关文件

- 模型定义: `models/T3Time_FreTS_Gated_Qwen_FusionExp.py`
- 训练脚本: `train_frets_gated_qwen_fusion_exp.py`
- 批量实验: `scripts/run_fusion_exp.sh`
- 调试指南: `scripts/DEBUG_GUIDE_FreTS.md`
