# 消融实验结果分析脚本使用说明

## 功能

`analyze_ablation_results.py` 脚本用于分析消融实验的结果，生成详细的对比报告。

## 使用方法

### 基本用法

```bash
# 分析默认结果文件 (experiment_results.log)
python scripts/T3Time_FreTS_FusionExp/analyze_ablation_results.py
```

### 指定结果文件

```bash
# 分析指定的结果文件
python scripts/T3Time_FreTS_FusionExp/analyze_ablation_results.py --result_file /path/to/results.log
```

### 导出 CSV

```bash
# 分析并导出到 CSV
python scripts/T3Time_FreTS_FusionExp/analyze_ablation_results.py --export_csv

# 指定 CSV 输出路径
python scripts/T3Time_FreTS_FusionExp/analyze_ablation_results.py --export_csv --csv_file ablation_results.csv
```

## 分析内容

脚本会分析以下消融实验：

### 1. 实验1: FreTS Component 的有效性
- **1.1** 固定 FFT vs FreTS Component
- **1.2** 仅幅度 vs 复数信息 (固定FFT)
- **1.3** 有无稀疏化机制 (FreTS)

### 2. 实验2: 融合机制对比
- Gate
- Weighted
- Cross Attention
- Hybrid

### 3. 实验3: 超参数敏感性分析
- **3.1** Scale 参数敏感性
- **3.2** Sparsity Threshold 参数敏感性

### 4. 实验4: 门控机制改进的影响
- 原始门控 vs 改进门控

### 5. 汇总表
- 所有实验的汇总对比

## 输出说明

### 统计信息

对于每个实验，脚本会输出：
- **最佳结果**: 所有运行中 MSE/MAE 最低的结果
- **均值 ± 标准差**: 多次运行的平均值和标准差
- **实验次数**: 该配置运行的次数

### 对比分析

对于对比实验（如 FreTS vs FFT），脚本会计算：
- **绝对改进**: `baseline_mse - improved_mse`
- **相对改进**: `(baseline_mse - improved_mse) / baseline_mse * 100%`

### 性能指标

- **MSE** (Mean Squared Error): 均方误差，越小越好
- **MAE** (Mean Absolute Error): 平均绝对误差，越小越好

## 示例输出

```
================================================================================
实验1: FreTS Component 的有效性
================================================================================

[1.1] 固定 FFT vs FreTS Component:
--------------------------------------------------------------------------------
  固定 FFT (Magnitude):
    最佳 MSE: 0.378939, MAE: 0.392907
    均值: 0.378939 ± 0.000000 (n=1)
  FreTS Component:
    最佳 MSE: 0.377179, MAE: 0.393050
    均值: 0.378088 ± 0.001286 (n=3)

  对比:
    FreTS 相对 FFT 改进: 0.001760 (+0.46%)
    ✅ FreTS 性能优于 FFT
```

## CSV 导出

导出的 CSV 文件包含以下字段：
- `ablation_exp`: 实验名称
- `fusion_mode`: 融合模式
- `use_frets`: 是否使用 FreTS
- `use_complex`: 是否使用复数
- `use_sparsity`: 是否使用稀疏化
- `use_improved_gate`: 是否使用改进门控
- `frets_scale`: FreTS scale 参数
- `sparsity_threshold`: 稀疏化阈值
- `test_mse`: 测试 MSE
- `test_mae`: 测试 MAE
- `seed`: 随机种子
- `timestamp`: 时间戳

## 注意事项

1. **结果文件格式**: 脚本期望结果文件是 JSON Lines 格式（每行一个 JSON 对象）
2. **实验标识**: 脚本通过 `model_id` 中包含 "Ablation" 或 `ablation_exp` 字段来识别消融实验结果
3. **多次运行**: 如果同一配置运行了多次（不同 seed），脚本会计算统计信息（均值、标准差等）
4. **最佳结果**: 脚本使用 MSE 作为主要指标来选择最佳结果

## 故障排除

### 未找到结果

如果脚本提示"未找到消融实验结果"，请检查：
1. 结果文件路径是否正确
2. 结果文件中是否包含消融实验的数据（`model_id` 包含 "Ablation" 或 `ablation_exp` 字段）

### JSON 解析错误

如果出现 JSON 解析错误，可能是结果文件格式有问题。请检查：
1. 每行是否是一个完整的 JSON 对象
2. JSON 格式是否正确
