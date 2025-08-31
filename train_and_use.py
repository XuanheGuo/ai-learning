import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math_model import MathPatternModel, MathDataset, train_model, analyze_patterns

def main():
    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型参数
    hidden_size = 1024
    num_layers = 12
    matrix_dim = 512
    vector_dim = 256
    
    # 创建模型
    print("创建数学规律发现模型...")
    model = MathPatternModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        matrix_dim=matrix_dim,
        vector_dim=vector_dim
    )
    
    # 创建数据集
    print("创建数据集...")
    dataset = MathDataset(matrix_dim=matrix_dim, vector_dim=vector_dim)
    
    # 训练模型
    print("开始训练模型...")
    trained_model = train_model(
        model=model,
        dataset=dataset,
        num_epochs=50,
        batch_size=16,
        device=device
    )
    
    # 保存模型
    torch.save(trained_model.state_dict(), 'math_pattern_model.pth')
    print("模型已保存到 math_pattern_model.pth")
    
    # 测试模型
    print("测试模型...")
    test_matrices = torch.randn(4, matrix_dim, matrix_dim)
    test_vectors = torch.randn(4, vector_dim)
    
    results = analyze_patterns(trained_model, test_matrices, test_vectors, device)
    
    print("模式分析结果:")
    print(f"生成的模式: {results['generated_patterns']}")
    print(f"模式概率分布形状: {results['pattern_probabilities'].shape}")
    print(f"关系概率分布形状: {results['relation_probabilities'].shape}")
    
    # 可视化结果
    visualize_results(results)
    
    return trained_model, results

def visualize_results(results):
    """可视化分析结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 模式概率分布
    pattern_probs = results['pattern_probabilities'].cpu().numpy()
    sns.heatmap(pattern_probs, ax=axes[0,0], cmap='viridis')
    axes[0,0].set_title('模式概率分布')
    axes[0,0].set_xlabel('模式类型')
    axes[0,0].set_ylabel('样本')
    
    # 关系概率分布
    relation_probs = results['relation_probabilities'].cpu().numpy()
    sns.heatmap(relation_probs, ax=axes[0,1], cmap='plasma')
    axes[0,1].set_title('关系概率分布')
    axes[0,1].set_xlabel('关系类型')
    axes[0,1].set_ylabel('样本')
    
    # 隐藏状态可视化
    hidden_states = results['hidden_states'].cpu().numpy()
    sns.heatmap(hidden_states[0], ax=axes[1,0], cmap='coolwarm')
    axes[1,0].set_title('隐藏状态 (第一个样本)')
    
    # 生成的模式
    generated_patterns = results['generated_patterns'].cpu().numpy()
    axes[1,1].bar(range(len(generated_patterns)), generated_patterns.flatten())
    axes[1,1].set_title('生成的模式')
    axes[1,1].set_xlabel('样本')
    axes[1,1].set_ylabel('模式类型')
    
    plt.tight_layout()
    plt.savefig('math_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("可视化结果已保存到 math_pattern_analysis.png")

def analyze_lattice_cryptography(model, device):
    """专门用于格密码学分析的函数"""
    print("进行格密码学特定分析...")
    
    # 生成格密码学相关的测试数据
    matrix_dim = 512
    vector_dim = 256
    
    # 生成随机矩阵（模拟格基）
    lattice_bases = torch.randn(4, matrix_dim, matrix_dim)
    
    # 生成随机向量（模拟消息或噪声）
    messages = torch.randn(4, vector_dim)
    
    # 分析
    results = analyze_patterns(model, lattice_bases, messages, device)
    
    print("格密码学分析结果:")
    print(f"格基矩阵形状: {lattice_bases.shape}")
    print(f"消息向量形状: {messages.shape}")
    print(f"发现的模式: {results['generated_patterns']}")
    
    return results

if __name__ == "__main__":
    # 运行主程序
    model, results = main()
    
    # 进行格密码学特定分析
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lattice_results = analyze_lattice_cryptography(model, device)
