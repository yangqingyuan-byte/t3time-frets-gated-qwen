# FreTS FusionExp 最小改动改进方案

## 🎯 原则：奥卡姆剃刀
- 不增加新架构
- 不改变模型结构
- 只调整训练策略和参数

## 🔍 发现的差异

### 1. 损失函数
- **T3Time V30**: `MSELoss`
- **FreTS FusionExp**: `SmoothL1Loss(beta=0.2)`
- **影响**: 损失函数的差异可能导致优化目标不同

### 2. 学习率调度器
- **T3Time V30**: `adjust_learning_rate` (Step Decay, type1)
- **FreTS FusionExp**: `adjust_learning_rate` (但可能没有正确调用)
- **影响**: 学习率衰减策略影响收敛

### 3. 归一化
- **T3Time V30**: `affine=False`
- **FreTS FusionExp**: `affine=True`
- **影响**: 归一化方式可能影响特征分布

### 4. FreTS Component 参数
- **sparsity_threshold**: 0.01 (可能过强，导致信息丢失)
- **scale**: 0.02 (初始化缩放)
- **影响**: 这些参数直接影响频域特征提取

## 💡 改进方案（按优先级）

### 优先级 1: 对齐训练策略（最简单）

#### 1.1 改用 MSELoss
```python
# 从
criterion = nn.SmoothL1Loss(beta=0.2)
# 改为
criterion = nn.MSELoss()
```

#### 1.2 确保使用 Step Decay
```python
# 确保调用
adjust_learning_rate(optimizer, epoch + 1, args)
```

#### 1.3 归一化改为 affine=False
```python
# 在模型定义中
self.normalize_layers = Normalize(num_nodes, affine=False).to(device)
```

### 优先级 2: 调整 FreTS Component 参数

#### 2.1 降低稀疏化阈值
```python
# 从 sparsity_threshold=0.01 改为 0.005 或 0.001
# 减少信息丢失
```

#### 2.2 调整初始化 scale
```python
# 从 scale=0.02 改为 0.01 或 0.05
# 可能需要实验找到最佳值
```

### 优先级 3: 微调融合机制参数

#### 3.1 调整 horizon_info 的权重
```python
# 当前: horizon_info = pred_len / 100.0
# 可以尝试: pred_len / 50.0 或 pred_len / 200.0
```

## 📝 实施步骤

### 步骤 1: 修改训练脚本
1. 改用 MSELoss
2. 确保使用 Step Decay
3. 添加 lradj 参数

### 步骤 2: 修改模型定义
1. 归一化改为 affine=False
2. 调整 FreTS Component 参数

### 步骤 3: 实验验证
1. 先改训练策略（步骤1）
2. 如果还不够，再改模型参数（步骤2）
