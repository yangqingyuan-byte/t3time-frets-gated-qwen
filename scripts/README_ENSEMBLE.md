# 集成学习使用说明

## 快速开始

### 方式1：使用3个种子（默认）

```bash
# 训练3个模型
bash scripts/run_ensemble_manual.sh

# 或单独运行评估（如果模型已训练好）
python scripts/eval_ensemble.py
```

### 方式2：使用20个种子

```bash
# 训练20个模型（从2024到2043）
bash scripts/run_ensemble_20seeds.sh

# 或单独运行评估（如果模型已训练好）
python scripts/eval_ensemble.py --start_seed 2024 --num_seeds 20
```

## 评估脚本参数说明

`eval_ensemble.py` 支持多种种子指定方式：

### 1. 使用默认种子（2024, 2025, 2026）
```bash
python scripts/eval_ensemble.py
```

### 2. 指定逗号分隔的种子列表
```bash
python scripts/eval_ensemble.py --seeds "2024,2025,2026,2027,2028"
```

### 3. 使用范围格式（从2024到2043，共20个）
```bash
python scripts/eval_ensemble.py --seeds "2024-2043"
```

### 4. 使用起始种子和数量（推荐，最灵活）
```bash
# 20个种子：从2024开始，共20个
python scripts/eval_ensemble.py --start_seed 2024 --num_seeds 20

# 10个种子：从2024开始，共10个
python scripts/eval_ensemble.py --start_seed 2024 --num_seeds 10
```

## 训练脚本说明

### `run_ensemble_manual.sh`（3个种子）
- 训练种子：2024, 2025, 2026
- 训练完成后自动运行集成评估

### `run_ensemble_20seeds.sh`（20个种子）
- 训练种子：2024 到 2043（共20个）
- 训练完成后自动运行集成评估

## 预期效果

- **3个种子集成**：MSE ≈ 0.383-0.385
- **20个种子集成**：MSE ≈ 0.382-0.384（理论上更稳定，可能更低）

## 注意事项

1. **训练时间**：20个模型需要约20倍的训练时间
2. **存储空间**：每个checkpoint约45MB，20个约900MB
3. **Checkpoint命名**：所有checkpoint保存在 `./checkpoints/checkpoint_seed_{seed}.pth`
4. **日志记录**：集成结果会自动写入 `experiment_results.log`

## 检查已训练的模型

```bash
# 查看所有checkpoint
ls -lh ./checkpoints/checkpoint_seed_*.pth

# 统计数量
ls ./checkpoints/checkpoint_seed_*.pth | wc -l
```
