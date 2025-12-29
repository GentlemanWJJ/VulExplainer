# 高级分类器使用指南

本文档介绍如何使用各种现代分类模型替代简单的线性分类器，以提升漏洞类型分类的性能。

## 可用的分类器类型

### 1. MLP分类器 (`mlp`) ⭐ 推荐
**优点：**
- 简单有效，训练稳定
- 适合大多数场景
- 计算效率高

**适用场景：** 通用漏洞分类任务

**使用示例：**
```python
model = AdvancedFusionModel(
    encoder=codebert,
    tokenizer=tokenizer,
    args=args,
    num_class=len(cwe_label_map),
    classifier_type='mlp'
)
```

### 2. 注意力分类器 (`attention`) ⭐⭐ 强烈推荐
**优点：**
- 自适应关注重要特征
- 能够学习特征之间的依赖关系
- 性能提升明显

**适用场景：** 需要关注特定代码模式的任务

**使用示例：**
```python
model = AdvancedFusionModel(
    encoder=codebert,
    tokenizer=tokenizer,
    args=args,
    num_class=len(cwe_label_map),
    classifier_type='attention'
)
```

### 3. 残差分类器 (`residual`)
**优点：**
- 通过残差连接缓解梯度消失
- 适合深层网络
- 训练更稳定

**适用场景：** 需要深层网络的复杂任务

### 4. Transformer分类器 (`transformer`) ⭐⭐⭐ 最推荐
**优点：**
- 强大的特征提取能力
- 能够捕获长距离依赖
- 在复杂任务上表现优异

**适用场景：** 复杂漏洞模式识别

**使用示例：**
```python
model = AdvancedFusionModel(
    encoder=codebert,
    tokenizer=tokenizer,
    args=args,
    num_class=len(cwe_label_map),
    classifier_type='transformer'
)
```

### 5. 标签感知分类器 (`label_aware`) ⭐⭐ 强烈推荐
**优点：**
- 通过标签嵌入增强分类性能
- 特别适合多分类任务
- 能够学习标签之间的关系

**适用场景：** CWE多分类任务（推荐）

**使用示例：**
```python
model = AdvancedFusionModel(
    encoder=codebert,
    tokenizer=tokenizer,
    args=args,
    num_class=len(cwe_label_map),
    classifier_type='label_aware'
)
```

### 6. 对比学习分类器 (`contrastive`)
**优点：**
- 提升特征表示质量
- 增强类间区分度
- 适合数据不平衡场景

**适用场景：** 类别不平衡的漏洞数据集

### 7. 层次化分类器 (`hierarchical`)
**优点：**
- 适合有层次结构的分类任务
- 能够利用CWE的层次关系

**适用场景：** CWE分类（有层次结构）

### 8. 胶囊网络分类器 (`capsule`)
**优点：**
- 能够捕获特征之间的空间关系
- 适合复杂模式识别

**适用场景：** 需要捕获代码结构关系的任务

## 在teacher_main.py中使用

修改 `teacher_main.py` 中的模型初始化部分：

```python
# 原始代码
# model = FusionModel(
#     encoder=codebert,
#     tokenizer=tokenizer,
#     args=args,
#     num_class=len(cwe_label_map)
# )

# 使用高级分类器
from model_advanced import AdvancedFusionModel

model = AdvancedFusionModel(
    encoder=codebert,
    tokenizer=tokenizer,
    args=args,
    num_class=len(cwe_label_map),
    classifier_type='attention'  # 或 'mlp', 'transformer', 'label_aware' 等
)
```

## 性能对比建议

建议按以下顺序尝试：

1. **首先尝试：** `mlp` - 简单有效，作为baseline
2. **其次尝试：** `attention` - 通常有2-5%的性能提升
3. **深度优化：** `transformer` 或 `label_aware` - 可能获得5-10%的性能提升
4. **特殊场景：** 
   - 数据不平衡 → `contrastive`
   - CWE层次结构 → `hierarchical`
   - 复杂模式 → `capsule`

## 超参数调优建议

### MLP分类器
```python
MLPClassifier(
    input_dim=256,
    hidden_dims=[256, 128],  # 可以调整层数和维度
    num_classes=num_class,
    dropout=0.1,  # 可以尝试 0.1-0.3
    activation='gelu'  # 或 'relu', 'tanh'
)
```

### 注意力分类器
```python
AttentionClassifier(
    input_dim=256,
    num_classes=num_class,
    num_heads=8,  # 可以尝试 4, 8, 16
    dropout=0.1
)
```

### Transformer分类器
```python
TransformerClassifier(
    input_dim=256,
    num_classes=num_class,
    num_layers=2,  # 可以尝试 1-4
    num_heads=8,
    hidden_dim=256,
    dropout=0.1
)
```

## 实验建议

1. **对比实验：** 在相同数据集上测试不同分类器
2. **消融实验：** 分析各组件对性能的贡献
3. **超参数搜索：** 使用网格搜索或随机搜索优化超参数
4. **集成方法：** 可以尝试集成多个分类器的结果

## 注意事项

1. **计算资源：** Transformer和LabelAware分类器需要更多计算资源
2. **训练时间：** 复杂分类器需要更长的训练时间
3. **过拟合风险：** 复杂模型更容易过拟合，注意使用dropout和正则化
4. **数据量：** 复杂模型通常需要更多数据才能发挥优势

## 预期性能提升

根据经验，相比简单线性分类器：
- **MLP分类器：** 2-5% 提升
- **注意力分类器：** 3-7% 提升
- **Transformer分类器：** 5-10% 提升
- **标签感知分类器：** 4-8% 提升（多分类任务）

实际提升取决于数据集特性和超参数设置。

