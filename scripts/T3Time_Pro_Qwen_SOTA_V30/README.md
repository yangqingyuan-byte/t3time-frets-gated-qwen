# T3Time_Pro_Qwen_SOTA_V30 参数寻优脚本

## 脚本说明

`ETTh1.sh` 是针对 `T3Time_Pro_Qwen_SOTA_V30` 模型的参数寻优脚本，仿照 `T3Time_Wavelet_Packet_Gated_Qwen/ETTh1.sh` 的结构编写。

## 主要特点

1. **自动参数搜索**：对预定义的配置列表进行网格搜索
2. **多种子验证**：每个配置对 seed 2020-2040 逐一运行（共21个种子）
3. **结果记录**：自动解析训练日志并追加到 `experiment_results.log`
4. **日志保存**：每个训练任务的日志保存在 `Results/T3Time_Pro_Qwen_SOTA_V30/ETTh1/` 目录

## 配置参数

脚本会搜索以下参数组合：

- **pred_len**: 96, 192, 336, 720
- **learning_rate**: 1e-4, 5e-5
- **channel**: 128, 256
- **dropout_n**: 0.3, 0.4, 0.5, 0.6
- **weight_decay**: 1e-3 (固定)
- **batch_size**: 16 (固定)
- **epochs**: 100 (固定)

## 使用方法

### 1. 直接运行（前台）
```bash
bash scripts/T3Time_Pro_Qwen_SOTA_V30/ETTh1.sh
```

### 2. 后台运行（推荐）
```bash
nohup bash scripts/T3Time_Pro_Qwen_SOTA_V30/ETTh1.sh > grid_search.log 2>&1 &
```

### 3. 查看进度
```bash
# 查看后台任务
tail -f grid_search.log

# 查看已完成的训练数量
ls Results/T3Time_Pro_Qwen_SOTA_V30/ETTh1/*.log | wc -l

# 查看最新的结果
tail -20 experiment_results.log
```

## 自定义配置

编辑 `ETTh1.sh` 中的 `CONFIGS` 数组来修改搜索空间：

```bash
CONFIGS=(
  # 格式: "pred_len lr channel dropout_n weight_decay batch_size epochs"
  "96 1e-4 128 0.5 1e-3 16 100"
  "96 5e-5 128 0.5 1e-3 16 100"
  # 添加更多配置...
)
```

## 结果分析

训练完成后，所有结果都记录在 `experiment_results.log` 中，格式为 JSONL（每行一个 JSON 对象）。

可以使用以下命令分析结果：

```bash
# 查看所有 V30 的结果
grep "T3Time_Pro_Qwen_SOTA_V30" experiment_results.log

# 按 MSE 排序，找出最佳配置
grep "T3Time_Pro_Qwen_SOTA_V30" experiment_results.log | \
  python -c "import sys, json; data=[json.loads(l) for l in sys.stdin]; \
  data.sort(key=lambda x: x['test_mse']); \
  [print(f\"MSE: {d['test_mse']:.6f}, Config: pred_len={d['pred_len']}, lr={d['learning_rate']}, c={d['channel']}, drop={d['dropout_n']}, seed={d['seed']}\") for d in data[:10]]"
```

## 注意事项

1. **训练时间**：每个配置 × 21个种子 = 大量训练任务，预计需要数天时间
2. **存储空间**：每个训练任务会产生日志文件，确保有足够空间
3. **GPU 使用**：脚本默认使用 GPU 1 (`CUDA_VISIBLE_DEVICES=1`)，可根据需要修改
4. **自动写入**：`train_learnable_wavelet_packet_pro.py` 会自动写入结果到日志，脚本中的解析作为备用

## 与原脚本的区别

1. 使用 `train_learnable_wavelet_packet_pro.py` 而不是 `train.py`
2. 使用 `--model_id` 和 `--model` 参数指定模型
3. 添加了 `--weight_decay` 和 `--wp_level` 参数
4. 移除了 `e_layer` 和 `d_layer`（V30 模型固定为 1 层）
5. 日志输出格式不同：`Pro Test Results: MSE: X, MAE: Y`
