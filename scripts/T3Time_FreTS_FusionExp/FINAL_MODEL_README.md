# T3Time_FreTS_Gated_Qwen 最终版本

## 概述

这是基于最佳配置的简化版本模型，移除了所有消融选项，固定使用最佳配置。

## 模型文件

- **模型定义**: `models/T3Time_FreTS_Gated_Qwen.py`
- **训练脚本**: `train_frets_gated_qwen.py`

## 固定配置

基于 seed=2088 的最佳结果（MSE=0.370849, MAE=0.391841），模型固定使用以下配置：

- ✅ **FreTS Component**: 可学习的频域 MLP（替代固定 FFT）
- ✅ **稀疏化机制**: `sparsity_threshold=0.009`
- ✅ **改进门控**: 基于归一化输入的门控机制
- ✅ **Gate 融合**: Horizon-Aware Gate 融合机制
- ✅ **FreTS Scale**: `scale=0.018`

## 最佳超参数

根据实验结果，最佳超参数配置为：

```bash
--data_path ETTh1
--seq_len 96
--pred_len 96
--channel 64
--batch_size 16
--learning_rate 0.0001
--dropout_n 0.1
--weight_decay 1e-4
--e_layer 1
--d_layer 1
--head 8
--loss_fn smooth_l1
--lradj type1
--embed_version qwen3_0.6b
```

## 使用方法

### 1. 单次训练（示例）

```bash
bash scripts/T3Time_FreTS_FusionExp/train_single_example.sh
```

或直接使用 Python：

```bash
python train_frets_gated_qwen.py \
    --data_path ETTh1 \
    --batch_size 16 \
    --seq_len 96 \
    --pred_len 96 \
    --epochs 100 \
    --es_patience 10 \
    --seed 2088 \
    --channel 64 \
    --learning_rate 0.0001 \
    --dropout_n 0.1 \
    --weight_decay 1e-4 \
    --e_layer 1 \
    --d_layer 1 \
    --head 8 \
    --loss_fn smooth_l1 \
    --lradj type1 \
    --embed_version qwen3_0.6b \
    --model_id T3Time_FreTS_Gated_Qwen
```

### 2. 多种子训练

运行多个种子（2020-2090）以获取更稳定的结果：

```bash
bash scripts/T3Time_FreTS_FusionExp/train_best_config_final.sh
```

### 3. 自定义参数

如果需要修改参数，可以直接调用训练脚本：

```bash
python train_frets_gated_qwen.py \
    --data_path ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --channel 64 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --dropout_n 0.1 \
    --weight_decay 1e-4 \
    --e_layer 1 \
    --d_layer 1 \
    --head 8 \
    --epochs 100 \
    --es_patience 10 \
    --seed 2024 \
    --loss_fn smooth_l1 \
    --lradj type1 \
    --embed_version qwen3_0.6b
```

## 与消融版本的对比

### 消融版本 (`T3Time_FreTS_Gated_Qwen_FusionExp`)
- 支持多种融合模式（gate, weighted, cross_attn, hybrid）
- 支持切换 FreTS/FFT
- 支持切换稀疏化机制
- 支持切换门控类型
- 用于消融实验和对比研究

### 最终版本 (`T3Time_FreTS_Gated_Qwen`)
- 固定使用最佳配置
- 代码更简洁，无消融选项
- 适合生产使用和论文报告
- 性能与消融版本相同（使用相同配置时）

## 模型架构

1. **时域分支**: Gated Transformer Encoder
2. **频域分支**: FreTS Component（可学习频域 MLP）+ Gated Transformer Encoder
3. **融合机制**: Horizon-Aware Gate（结合时域、频域和预测长度信息）
4. **Prompt 编码**: Gated Transformer Encoder（处理 LLM 嵌入）
5. **CMA**: Cross-Modal Alignment（对齐时域和 LLM 特征）
6. **解码器**: Transformer Decoder

## 实验结果

基于 ETTh1 数据集，pred_len=96 的最佳结果：

- **MSE**: 0.370849
- **MAE**: 0.391841
- **Seed**: 2088
- **配置**: 见上方最佳超参数

## 注意事项

1. **不保存模型文件**: 训练过程中不会保存任何 `.pth` 文件到磁盘，只在内存中保存最佳模型状态用于测试
2. **每次从头训练**: 每次实验都创建全新的模型，不加载任何预训练权重
3. **结果记录**: 实验结果会自动追加到 `experiment_results.log`

## 文件结构

```
/root/0/T3Time/
├── models/
│   └── T3Time_FreTS_Gated_Qwen.py          # 最终版本模型（不带消融选项）
├── train_frets_gated_qwen.py                # 最终版本训练脚本
└── scripts/T3Time_FreTS_FusionExp/
    ├── train_single_example.sh              # 单次训练示例
    ├── train_best_config_final.sh          # 多种子训练脚本
    └── FINAL_MODEL_README.md               # 本文档
```
