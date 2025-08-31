# Colab数学规律发现模型示例
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 检查TPU可用性
try:
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
    print("TPU可用")
except ImportError:
    TPU_AVAILABLE = False
    print("TPU不可用，使用CPU/GPU")

class MathPatternModel(nn.Module):
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
    matrices = torch.randn(num_samples, matrix_dim, matrix_dim)
    vectors = torch.randn(num_samples, vector_dim)
    symbols = torch.randint(0, 1000, (num_samples,))
    relations = torch.randint(0, 20, (num_samples,))
    
    return matrices, vectors, symbols, relations

def train_model(model, num_epochs=50, batch_size=32):
    """训练模型"""
    if TPU_AVAILABLE:
        device = xm.xla_device()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    pattern_loss_fn = nn.CrossEntropyLoss()
    relation_loss_fn = nn.CrossEntropyLoss()
    
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
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')
    
    return model

def analyze_patterns(model, matrices, vectors):
    """分析数学模式"""
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

# 主程序
def main():
    print("=== 数学规律发现大模型 ===")
    
    # 创建模型
    print("创建模型...")
    model = MathPatternModel(
        hidden_size=1024,
        num_layers=12,
        matrix_dim=512,
        vector_dim=256
    )
    
    # 训练模型
    print("开始训练...")
    trained_model = train_model(model, num_epochs=50, batch_size=32)
    
    # 测试模型
    print("测试模型...")
    test_matrices = torch.randn(4, 512, 512)
    test_vectors = torch.randn(4, 256)
    
    results = analyze_patterns(trained_model, test_matrices, test_vectors)
    
    print("分析完成!")
    print(f"模式概率分布形状: {results['pattern_probabilities'].shape}")
    print(f"关系概率分布形状: {results['relation_probabilities'].shape}")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(results['pattern_probabilities'].cpu().numpy(), cmap='viridis')
    plt.title('模式概率分布')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(results['relation_probabilities'].cpu().numpy(), cmap='plasma')
    plt.title('关系概率分布')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    return trained_model, results

if __name__ == "__main__":
    model, results = main()
