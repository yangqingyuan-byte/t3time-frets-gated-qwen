# T3Time_FreEformer_Gated_Qwen 参数对应关系说明

## 参数对应关系表

### ✅ 可以直接对应的参数

| FreEformer 参数 | T3Time_FreEformer_Gated_Qwen 参数 | 说明 | 推荐值 |
|----------------|----------------------------------|------|--------|
| `--embed_size` | `--embed_size` | 频域 token embedding 维度 | 16 |
| `--d_model` | `--d_model` | 模型维度（频域 Transformer） | 512 |
| `--d_ff` | `--d_ff` | 前馈网络维度 | 512 |
| `--e_layers` | `--fre_e_layer` | 频域 Transformer 编码器层数 | 2 |
| `--dropout` | `--dropout_n` | Dropout 比率 | 0.2 |
| `--batch_size` | `--batch_size` | 批次大小 | 32 |
| `--learning_rate` | `--learning_rate` | 学习率 | 1e-4 |
| `--train_epochs` | `--epochs` | 训练轮数 | 30 |
| `--patience` | `--es_patience` | Early stopping 耐心值 | 8 |
| `--lradj` | `--lradj` | 学习率调整策略 | type1 |
| `--seq_len` | `--seq_len` | 输入序列长度 | 96 |
| `--pred_len` | `--pred_len` | 预测长度 | 96 |
| `--enc_in` | `--num_nodes` | 输入特征数 | 7 |
| `--loss_mode L1` | `--loss_fn smooth_l1` | 损失函数（L1 对应 smooth_l1） | smooth_l1 |

### ⚠️ 需要转换的参数

| FreEformer 参数 | T3Time_FreEformer_Gated_Qwen 参数 | 转换说明 |
|----------------|----------------------------------|----------|
| `--attn_enhance 1` | `--attn_enhance 1` | 注意力增强模式（1=Enhanced, None/0=Vanilla） |
| `--attn_softmax_flag 0` | `--attn_softmax_flag 0` | 注意力 softmax 标志（0=False, 1=True） |
| `--attn_weight_plus 1` | `--attn_weight_plus 1` | 注意力权重加法模式（0=False, 1=True） |
| `--attn_outside_softmax 1` | `--attn_outside_softmax 1` | 注意力外部 softmax（0=False, 1=True） |

### ❌ 不适用或已固定的参数

| FreEformer 参数 | 说明 | 原因 |
|----------------|------|------|
| `--model FrePatchTST3_attn_ablation` | 模型名称 | T3Time_FreEformer_Gated_Qwen 是固定模型 |
| `--use_revin 1` | 使用 RevIN | T3Time 使用自己的 Normalize 层（affine=False） |
| `--use_norm 1` | 使用归一化 | T3Time 固定使用 Normalize |
| `--n_heads` | 注意力头数 | 使用 `--head` 参数（默认 8） |
| `--dec_in` | 解码器输入维度 | T3Time 架构不同，不使用解码器 |
| `--c_out` | 输出通道数 | 使用 `--num_nodes` |
| `--features M` | 特征模式 | T3Time 固定为多变量预测 |
| `--root_path` | 数据根路径 | 使用 `--data_path` 直接指定数据集名称 |

### 🔧 T3Time 特有的参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--channel` | 模型通道数（时域和频域共享） | 64 |
| `--e_layer` | 时域编码器层数 | 1 |
| `--d_layer` | 解码器层数（T3Time 架构） | 1 |
| `--head` | 注意力头数 | 8 |
| `--embed_version` | LLM 嵌入版本 | qwen3_0.6b |
| `--weight_decay` | 权重衰减 | 1e-4 |

## 推荐参数配置

### 基础配置（pred_len=96）

```bash
python -u train_freeformer_gated_qwen.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --num_nodes 7 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --dropout_n 0.2 \
  --channel 64 \
  --e_layer 1 \
  --d_layer 1 \
  --head 8 \
  --epochs 30 \
  --es_patience 8 \
  --lradj type1 \
  --embed_version qwen3_0.6b \
  --seed 2021 \
  --weight_decay 1e-4 \
  --loss_fn smooth_l1 \
  --model_id T3Time_FreEformer_Gated_Qwen_ETTh1_96 \
  --embed_size 16 \
  --fre_e_layer 2 \
  --d_model 512 \
  --d_ff 512 \
  --attn_enhance 1 \
  --attn_softmax_flag 0 \
  --attn_weight_plus 1 \
  --attn_outside_softmax 1
```

### 长序列配置（pred_len=720）

```bash
python -u train_freeformer_gated_qwen.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 720 \
  --num_nodes 7 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --dropout_n 0.2 \
  --channel 64 \
  --e_layer 1 \
  --d_layer 1 \
  --head 8 \
  --epochs 30 \
  --es_patience 8 \
  --lradj type1 \
  --embed_version qwen3_0.6b \
  --seed 2021 \
  --weight_decay 1e-4 \
  --loss_fn smooth_l1 \
  --model_id T3Time_FreEformer_Gated_Qwen_ETTh1_720 \
  --embed_size 16 \
  --fre_e_layer 2 \
  --d_model 512 \
  --d_ff 512 \
  --attn_enhance 1 \
  --attn_softmax_flag 0 \
  --attn_weight_plus 1 \
  --attn_outside_softmax 1
```

## 参数说明

### 注意力相关参数

- **`--attn_enhance 1`**: 启用增强注意力模式（SF_mode=1），使用可学习的权重矩阵
- **`--attn_softmax_flag 0`**: 不使用 softmax 对权重矩阵进行归一化，使用 softplus
- **`--attn_weight_plus 1`**: 使用加法模式（A = A + weight_mat）而非乘法模式
- **`--attn_outside_softmax 1`**: 在 softmax 外部应用权重矩阵

### 模型维度参数

- **`--d_model 512`**: 频域 Transformer 的模型维度（独立于 `--channel`）
- **`--d_ff 512`**: 频域 Transformer 的前馈网络维度（通常等于 d_model）
- **`--channel 64`**: T3Time 的主通道数（时域和频域共享）

### 架构差异说明

1. **归一化**: T3Time 使用 `Normalize` 层（`affine=False`），而 FreEformer 使用 RevIN（`affine=True`）
2. **融合机制**: T3Time 使用 Gate 融合机制，结合时域、频域和 LLM 嵌入
3. **频域处理**: T3Time 的频域处理在 FreEformer Component 中，输出后还会经过 Gated Transformer 和 Attention Pooling

## 使用建议

1. **首次训练**: 使用基础配置（pred_len=96）进行快速验证
2. **长序列**: 对于 pred_len=720，可能需要调整 `--fre_e_layer` 或 `--d_model`
3. **注意力参数**: 如果效果不佳，可以尝试：
   - `--attn_enhance 0` (Vanilla attention)
   - `--attn_softmax_flag 1` (使用 softmax)
   - `--attn_weight_plus 0` (使用乘法模式)
