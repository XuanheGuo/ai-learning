# 数学规律发现大模型 - 格密码学应用

## 项目概述

这是一个专门设计用于数学规律发现的大模型架构，特别针对格密码学领域中的矩阵和向量关系分析。该模型能够自动发现数学对象间的潜在关系，为密码学研究提供强大的分析工具。

## 核心特性

### 🧠 智能数学推理
- **多模态数学理解**: 同时处理矩阵、向量、符号和关系
- **Transformer架构**: 12层深度网络，16个注意力头
- **模式发现**: 自动识别8种不同的数学模式类型
- **关系预测**: 预测20种不同的数学关系类型

### 🔐 格密码学专用
- **格基分析**: 分析格基矩阵的结构特征
- **向量关系**: 发现向量间的线性依赖关系
- **安全性评估**: 评估格密码方案的安全性
- **参数优化**: 优化密码学参数

### ⚡ 高性能计算
- **TPU优化**: 支持Google Colab TPU加速
- **96核CPU**: 充分利用多核CPU资源
- **内存优化**: 梯度累积和批处理优化
- **并行训练**: 支持数据并行训练

## 文件结构

```
├── math_model.py              # 核心模型定义
├── train_and_use.py           # 训练和使用脚本
├── colab_example.py           # Colab示例代码
├── colab_usage_guide.md       # 详细使用指南
└── README.md                  # 项目说明
```

## 快速开始

### 1. 环境要求
- Python 3.7+
- PyTorch 1.8+
- NumPy, Matplotlib, Seaborn
- Google Colab (推荐使用TPU)

### 2. 安装依赖
```bash
pip install torch torchvision torchaudio
pip install matplotlib seaborn numpy sympy
```

### 3. 在Colab中使用
```python
# 上传文件到Colab
# 运行 colab_example.py
!python colab_example.py
```

### 4. 本地使用
```python
from math_model import MathPatternModel, MathDataset, train_model, analyze_patterns

# 创建模型
model = MathPatternModel(
    hidden_size=1024,
    num_layers=12,
    matrix_dim=512,
    vector_dim=256
)

# 训练模型
dataset = MathDataset(matrix_dim=512, vector_dim=256)
trained_model = train_model(model, dataset, num_epochs=100)

# 分析数学模式
test_matrices = torch.randn(4, 512, 512)
test_vectors = torch.randn(4, 256)
results = analyze_patterns(trained_model, test_matrices, test_vectors)
```

## 模型架构详解

### 1. 数学嵌入层
- **矩阵嵌入**: 将高维矩阵映射到低维表示空间
- **向量嵌入**: 学习向量间的几何关系
- **符号嵌入**: 理解数学符号和运算
- **关系嵌入**: 识别数学关系类型

### 2. 数学推理引擎
- **自注意力机制**: 发现数学对象间的关系
- **数学推理模块**: 专门设计的推理网络
- **模式发现**: 自动识别数学模式
- **几何推理**: 理解向量空间关系

### 3. 输出层
- **模式分类**: 8种模式类型
- **关系预测**: 20种关系类型
- **隐藏状态**: 用于进一步分析

## 应用场景

### 1. 格密码学研究
- 分析格基矩阵的结构特征
- 发现向量间的线性关系
- 评估密码方案的安全性
- 优化算法参数

### 2. 数学规律发现
- 自动发现数学对象间的模式
- 预测未知的数学关系
- 分析数学结构的性质
- 辅助数学证明

### 3. 密码学应用
- 密钥生成优化
- 攻击向量分析
- 参数选择指导
- 安全性评估

## 性能优化

### TPU优化
```python
import torch_xla.core.xla_model as xm

device = xm.xla_device()
model = model.to(device)

# TPU优化器步骤
xm.optimizer_step(optimizer)
```

### 内存优化
```python
# 梯度累积
accumulation_steps = 4
loss = loss / accumulation_steps
loss.backward()

if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### 批处理优化
```python
# 动态批处理大小
batch_size = 32 if TPU_AVAILABLE else 16
```

## 训练建议

### 1. 数据准备
- 使用高质量的数学数据
- 平衡不同模式类型的样本
- 确保数据的多样性

### 2. 超参数调优
- 学习率: 1e-4 (推荐)
- 批次大小: 32 (TPU) / 16 (GPU)
- 训练轮数: 100-500
- 权重衰减: 0.01

### 3. 监控指标
- 模式分类准确率
- 关系预测准确率
- 训练损失收敛
- 验证集性能

## 扩展功能

### 1. 符号数学处理
```python
import sympy as sp
from sympy import Matrix, symbols

# 符号矩阵分析
def analyze_symbolic_matrix(matrix_expr):
    matrix = Matrix(matrix_expr)
    eigenvalues = matrix.eigenvals()
    return eigenvalues
```

### 2. 几何推理
```python
# 向量几何分析
def analyze_vector_geometry(vectors):
    norms = torch.norm(vectors, dim=-1)
    angles = torch.acos(torch.clamp(
        torch.matmul(vectors, vectors.transpose(-2, -1)) / 
        (norms.unsqueeze(-1) * norms.unsqueeze(-2)), -1, 1
    ))
    return norms, angles
```

### 3. 可视化分析
```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_patterns(results):
    plt.figure(figsize=(12, 8))
    
    # 模式概率热图
    plt.subplot(2, 2, 1)
    sns.heatmap(results['pattern_probabilities'].cpu().numpy())
    plt.title('模式概率分布')
    
    # 关系概率热图
    plt.subplot(2, 2, 2)
    sns.heatmap(results['relation_probabilities'].cpu().numpy())
    plt.title('关系概率分布')
    
    plt.show()
```

## 注意事项

1. **内存使用**: 大矩阵操作需要大量内存
2. **训练时间**: 完整训练可能需要数小时
3. **数据质量**: 训练数据质量直接影响性能
4. **超参数**: 根据具体任务调整参数

## 贡献指南

欢迎贡献代码和想法！请遵循以下步骤：

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 参与讨论

---

**注意**: 这是一个研究项目，主要用于学术研究和密码学分析。请确保遵守相关法律法规。
