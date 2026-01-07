# T3Time_FreTS_FusionExp ETTh1 参数寻优脚本

## 📋 脚本说明

`ETTh1_hyperopt.sh` 是针对 `T3Time_FreTS_FusionExp` 模型的参数寻优脚本，参考 `T3Time_Wavelet_Packet_Gated_Qwen` 的格式编写。

## 🎯 固定参数（基于最佳配置）

基于参数寻优结果，以下参数已固定为最佳值：
- **Scale**: 0.018
- **Sparsity Threshold**: 0.009
- **Fusion Mode**: gate
- **Loss Function**: smooth_l1
- **Weight Decay**: 1e-4
- **Dropout**: 0.1（基础配置）

## 🔬 寻优参数

脚本会对以下参数进行网格搜索：

| 参数 | 说明 | 测试范围 |
|------|------|---------|
| **pred_len** | 预测长度 | 96, 192, 336, 720 |
| **learning_rate** | 学习率 | 1e-4, 5e-5, 7e-5 |
| **channel** | 特征维度 | 64, 128 |
| **e_layer** | 编码器层数 | 1, 2 |
| **d_layer** | 解码器层数 | 1, 2, 3, 4 |
| **batch_size** | 批次大小 | 16, 32 |
| **epochs** | 训练轮数 | 100, 120, 150 |

## 📊 测试配置列表

### pred_len = 96
- `96 1e-4 64 1 1 0.1 1e-4 16 100`
- `96 1e-4 128 1 1 0.1 1e-4 16 100`
- `96 5e-5 64 1 1 0.1 1e-4 16 100`
- `96 1e-4 64 2 1 0.1 1e-4 16 100`
- `96 1e-4 64 1 2 0.1 1e-4 16 100`

### pred_len = 192
- `192 1e-4 64 1 1 0.1 1e-4 16 100`
- `192 1e-4 128 1 1 0.1 1e-4 16 100`
- `192 5e-5 64 1 2 0.1 1e-4 16 100`
- `192 1e-4 64 2 2 0.1 1e-4 16 100`

### pred_len = 336
- `336 1e-4 64 1 1 0.1 1e-4 16 100`
- `336 1e-4 128 1 2 0.1 1e-4 16 100`
- `336 5e-5 64 1 2 0.1 1e-4 16 100`

### pred_len = 720
- `720 1e-4 64 1 2 0.1 1e-4 16 100`
- `720 1e-4 128 2 2 0.1 1e-4 16 100`
- `720 5e-5 64 2 2 0.1 1e-4 16 100`

**总计**: 15 个配置组合，每个配置对 seed 2020-2040 运行（21 个种子）

## 🚀 使用方法

### 1. 直接运行（前台）

```bash
bash scripts/T3Time_FreTS_FusionExp/ETTh1_hyperopt.sh
```

### 2. 后台运行（推荐）

```bash
nohup bash scripts/T3Time_FreTS_FusionExp/ETTh1_hyperopt.sh > frets_etth1_hyperopt.log 2>&1 &
```

### 3. 查看进度

```bash
# 查看后台任务
tail -f frets_etth1_hyperopt.log

# 查看已完成的训练数量
ls Results/T3Time_FreTS_FusionExp/ETTh1/*.log | wc -l

# 查看最新的结果
tail -20 experiment_results.log
```

## 📈 结果分析

训练完成后，所有结果都记录在 `experiment_results.log` 中。

### 快速查看最佳结果

```bash
# 查看所有 FreTS FusionExp 的结果
grep "T3Time_FreTS_FusionExp" experiment_results.log | \
  python -c "
import sys, json
results = []
for line in sys.stdin:
    data = json.loads(line.strip())
    if data.get('pred_len') in [96, 192, 336, 720]:
        results.append((
            data.get('pred_len', 0),
            data.get('channel', 0),
            data.get('e_layer', 0),
            data.get('d_layer', 0),
            data.get('learning_rate', 0),
            data['test_mse'],
            data['test_mae'],
            data.get('seed', 'unknown')
        ))
results.sort(key=lambda x: (x[0], x[5]))  # 按 pred_len 和 MSE 排序

print('='*80)
print('T3Time_FreTS_FusionExp 参数寻优结果（按 pred_len 和 MSE 排序）')
print('='*80)
print(f\"{'Pred':<6} {'Channel':<8} {'E_Layer':<8} {'D_Layer':<8} {'LR':<10} {'MSE':<12} {'MAE':<12} {'Seed':<8}\")
print('-'*80)
for pred, ch, el, dl, lr, mse, mae, seed in results[:20]:
    print(f'{pred:<6} {ch:<8} {el:<8} {dl:<8} {lr:<10.0e} {mse:<12.6f} {mae:<12.6f} {seed:<8}')
"
```

### 使用筛选脚本

```bash
python 筛选分析实验结果.py
# 然后选择: T3Time_FreTS_FusionExp
```

## ⚙️ 自定义配置

编辑 `ETTh1_hyperopt.sh` 中的 `CONFIGS` 数组来修改搜索空间：

```bash
CONFIGS=(
  # 格式: "pred_len lr channel e_layer d_layer dropout_n weight_decay batch_size epochs"
  "96 1e-4 64 1 1 0.1 1e-4 16 100"
  # 添加更多配置...
)
```

## 📝 注意事项

1. **训练时间**：每个配置会运行 21 个种子（2020-2040），总训练时间较长
2. **建议使用后台运行**，避免终端断开
3. **结果会自动追加**到 `experiment_results.log`
4. **日志文件保存在** `Results/T3Time_FreTS_FusionExp/ETTh1/`
5. **固定参数**：scale=0.018, sparsity_threshold=0.009 已固定为最佳值

## 🔧 故障排除

如果遇到问题：

1. **CUDA 内存不足**: 调整 `CUDA_VISIBLE_DEVICES` 或减少并行任务
2. **日志解析失败**: 检查训练是否正常完成
3. **结果未写入**: 检查 `experiment_results.log` 的写入权限

## 📊 预期结果

基于当前最佳配置（pred_len=96, MSE=0.376336），预期：
- **pred_len=96**: 可能找到 MSE < 0.376 的配置
- **pred_len=192/336/720**: 探索不同预测长度下的最佳超参数组合
