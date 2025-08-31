import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    
    def generate_patterns(self, matrices, vectors, symbols, relations, temperature=0.8):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(matrices, vectors, symbols, relations)
            pattern_probs = F.softmax(outputs['pattern_logits'] / temperature, dim=-1)
            generated_patterns = torch.multinomial(pattern_probs, 1)
        return generated_patterns, outputs

class MathDataset:
    def __init__(self, matrix_dim=512, vector_dim=256):
        self.matrix_dim = matrix_dim
        self.vector_dim = vector_dim
    
    def generate_data(self, num_samples=1000):
        matrices = torch.randn(num_samples, self.matrix_dim, self.matrix_dim)
        vectors = torch.randn(num_samples, self.vector_dim)
        symbols = torch.randint(0, 1000, (num_samples,))
        relations = torch.randint(0, 20, (num_samples,))
        
        return matrices, vectors, symbols, relations
    
    def generate_labels(self, matrices, vectors):
        relation_labels = torch.randint(0, 20, (matrices.shape[0],))
        pattern_labels = torch.randint(0, 8, (matrices.shape[0],))
        return relation_labels, pattern_labels

def train_model(model, dataset, num_epochs=100, batch_size=32, device='cuda'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    pattern_loss_fn = nn.CrossEntropyLoss()
    relation_loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        
        matrices, vectors, symbols, relations = dataset.generate_data(batch_size)
        relation_labels, pattern_labels = dataset.generate_labels(matrices, vectors)
        
        matrices = matrices.to(device)
        vectors = vectors.to(device)
        symbols = symbols.to(device)
        relations = relations.to(device)
        relation_labels = relation_labels.to(device)
        pattern_labels = pattern_labels.to(device)
        
        outputs = model(matrices, vectors, symbols, relations)
        
        pattern_loss = pattern_loss_fn(outputs['pattern_logits'], pattern_labels)
        relation_loss = relation_loss_fn(outputs['relation_output'], relation_labels)
        total_loss = pattern_loss + relation_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')
    
    return model

def analyze_patterns(model, matrices, vectors, device='cuda'):
    model.eval()
    with torch.no_grad():
        batch_size = matrices.shape[0]
        symbols = torch.randint(0, 1000, (batch_size,)).to(device)
        relations = torch.randint(0, 20, (batch_size,)).to(device)
        
        matrices = matrices.to(device)
        vectors = vectors.to(device)
        
        generated_patterns, outputs = model.generate_patterns(
            matrices, vectors, symbols, relations
        )
        
        pattern_probs = F.softmax(outputs['pattern_logits'], dim=-1)
        relation_probs = F.softmax(outputs['relation_output'], dim=-1)
        
        return {
            'generated_patterns': generated_patterns,
            'pattern_probabilities': pattern_probs,
            'relation_probabilities': relation_probs,
            'hidden_states': outputs['hidden_states']
        }
