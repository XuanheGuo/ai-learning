#!/usr/bin/env python3
"""
数学规律发现大模型 - 快速开始示例
适用于Google Colab + TPU环境
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 检查TPU
try:
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
    print("✅ TPU可用")
except ImportError:
    TPU_AVAILABLE = False
    print("⚠️  TPU不可用，使用CPU/GPU")

class MathPatternModel(nn.Module):
    """数学规律发现模型"""
    
    def __init__(self, hidden_size=1024, num_layers=12, matrix_dim=512, vector_dim=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.matrix_dim = matrix_dim
        self.vector_dim = vector_dim
        
        # 嵌入层
        self.matrix_embedding = nn.Linear(matrix_dim * matrix_dim, hidden_size)
        self.vector_embedding = nn.Linear(vector_dim, hidden_size)
        self.symbol_embedding = nn.Embedding(1000, hidden_size)
        self.relation_embedding = nn.Embedding(20, hidden_size)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=16,
            dim_feedforward=4096,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 模式发现
        self.pattern_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 8)
        )
        
        # 输出层
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.relation_predictor = nn.Linear(hidden_size, 20)
        
    def forward(self, matrices, vectors, symbols, relations):
        batch_size, seq_len = matrices.shape[:2]
        
        # 矩阵嵌入
        matrices_flat = matrices.view(batch_size, seq_len, -1)
        matrix_emb = self.matrix_embedding(matrices_flat)
        
        # 向量嵌入
        vector_emb = self.vector_embedding(vectors)
        
        # 符号和关系嵌入
        symbol_emb = self.symbol_embedding(symbols)
        relation_emb = self.relation_embedding(relations)
        
        # 组合嵌入
        combined_emb = matrix_emb + vector_emb + symbol_emb + relation_emb
        
        # Transformer处理
        hidden_states = self.transformer(combined_emb.transpose(0, 1)).transpose(0, 1)
        
        # 模式发现
        pattern_logits = self.pattern_encoder(hidden_states)
        
        # 输出
        output = self.output_projection(hidden_states)
        relation_output = self.relation_predictor(hidden_states)
        
        return {
            'hidden_states': output,
            'pattern_logits': pattern_logits,
            'relation_output': relation_output
        }

def generate_lattice_data(num_samples=1000, matrix_dim=512, vector_dim=256):
    """生成格密码学数据"""
    print(f"生成 {num_samples} 个样本...")
    
    # 生成随机矩阵（模拟格基）
    matrices = torch.randn(num_samples, matrix_dim, matrix_dim)
    
    # 生成随机向量（模拟消息）
    vectors = torch.randn(num_samples, vector_dim)
    
    # 生成数学符号
    symbols = torch.randint(0, 1000, (num_samples,))
    
    # 生成关系类型
    relations = torch.randint(0, 20, (num_samples,))
    
    return matrices, vectors, symbols, relations

def train_model(model, num_epochs=50, batch_size=32):
    """训练模型"""
    print("开始训练模型...")
    
    # 选择设备
    if TPU_AVAILABLE:
        device = xm.xla_device()
        print("使用TPU训练")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    pattern_loss_fn = nn.CrossEntropyLoss()
    relation_loss_fn = nn.CrossEntropyLoss()
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        
        # 生成训练数据
        matrices, vectors, symbols, relations = generate_lattice_data(batch_size)
        relation_labels = torch.randint(0, 20, (batch_size,))
        pattern_labels = torch.randint(0, 8, (batch_size,))
        
        # 移动到设备
        matrices = matrices.to(device)
        vectors = vectors.to(device)
        symbols = symbols.to(device)
        relations = relations.to(device)
        relation_labels = relation_labels.to(device)
        pattern_labels = pattern_labels.to(device)
        
        # 前向传播
        outputs = model(matrices, vectors, symbols, relations)
        
        # 计算损失
        pattern_loss = pattern_loss_fn(outputs['pattern_logits'], pattern_labels)
        relation_loss = relation_loss_fn(outputs['relation_output'], relation_labels)
        total_loss = pattern_loss + relation_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        if TPU_AVAILABLE:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        
        losses.append(total_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    return model

def analyze_patterns(model, matrices, vectors):
    """分析数学模式"""
    print("分析数学模式...")
    
    if TPU_AVAILABLE:
        device = xm.xla_device()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    with torch.no_grad():
        batch_size = matrices.shape[0]
        symbols = torch.randint(0, 1000, (batch_size,)).to(device)
        relations = torch.randint(0, 20, (batch_size,)).to(device)
        
        matrices = matrices.to(device)
        vectors = vectors.to(device)
        
        outputs = model(matrices, vectors, symbols, relations)
        
        pattern_probs = F.softmax(outputs['pattern_logits'], dim=-1)
        relation_probs = F.softmax(outputs['relation_output'], dim=-1)
        
        return {
            'pattern_probabilities': pattern_probs,
            'relation_probabilities': relation_probs,
            'hidden_states': outputs['hidden_states']
        }

def visualize_results(results):
    """可视化分析结果"""
    print("生成可视化结果...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 模式概率分布
    pattern_probs = results['pattern_probabilities'].cpu().numpy()
    im1 = axes[0,0].imshow(pattern_probs, cmap='viridis', aspect='auto')
    axes[0,0].set_title('模式概率分布', fontsize=14)
    axes[0,0].set_xlabel('模式类型')
    axes[0,0].set_ylabel('样本')
    plt.colorbar(im1, ax=axes[0,0])
    
    # 关系概率分布
    relation_probs = results['relation_probabilities'].cpu().numpy()
    im2 = axes[0,1].imshow(relation_probs, cmap='plasma', aspect='auto')
    axes[0,1].set_title('关系概率分布', fontsize=14)
    axes[0,1].set_xlabel('关系类型')
    axes[0,1].set_ylabel('样本')
    plt.colorbar(im2, ax=axes[0,1])
    
    # 隐藏状态可视化
    hidden_states = results['hidden_states'].cpu().numpy()
    im3 = axes[1,0].imshow(hidden_states[0], cmap='coolwarm', aspect='auto')
    axes[1,0].set_title('隐藏状态 (第一个样本)', fontsize=14)
    axes[1,0].set_xlabel('隐藏维度')
    axes[1,0].set_ylabel('序列长度')
    plt.colorbar(im3, ax=axes[1,0])
    
    # 模式分布统计
    pattern_means = pattern_probs.mean(axis=0)
    axes[1,1].bar(range(len(pattern_means)), pattern_means, color='skyblue')
    axes[1,1].set_title('平均模式分布', fontsize=14)
    axes[1,1].set_xlabel('模式类型')
    axes[1,1].set_ylabel('平均概率')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("可视化完成!")

def main():
    """主函数"""
    print("=" * 60)
    print("🧠 数学规律发现大模型 - 格密码学应用")
    print("=" * 60)
    
    # 1. 创建模型
    print("\n1️⃣ 创建模型...")
    model = MathPatternModel(
        hidden_size=1024,
        num_layers=12,
        matrix_dim=512,
        vector_dim=256
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 训练模型
    print("\n2️⃣ 训练模型...")
    trained_model = train_model(model, num_epochs=50, batch_size=32)
    
    # 3. 测试模型
    print("\n3️⃣ 测试模型...")
    test_matrices = torch.randn(4, 512, 512)
    test_vectors = torch.randn(4, 256)
    
    results = analyze_patterns(trained_model, test_matrices, test_vectors)
    
    # 4. 可视化结果
    print("\n4️⃣ 可视化结果...")
    visualize_results(results)
    
    # 5. 输出统计信息
    print("\n5️⃣ 分析统计:")
    print(f"模式概率分布形状: {results['pattern_probabilities'].shape}")
    print(f"关系概率分布形状: {results['relation_probabilities'].shape}")
    print(f"隐藏状态形状: {results['hidden_states'].shape}")
    
    print("\n✅ 分析完成!")
    print("=" * 60)
    
    return trained_model, results

if __name__ == "__main__":
    model, results = main()
