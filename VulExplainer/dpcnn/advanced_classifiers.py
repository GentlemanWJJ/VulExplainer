"""
高级分类器模块 - 用于漏洞类型分类
提供多种现代分类模型，可替代简单的线性分类器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLPClassifier(nn.Module):
    """
    多层感知机分类器
    优点：简单有效，适合中等复杂度任务
    """
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1, activation='relu'):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
    def forward(self, x):
        features = self.layers(x)
        logits = self.classifier(features)
        return logits


class AttentionClassifier(nn.Module):
    """
    注意力机制增强的分类器
    优点：能够自适应地关注重要特征，提升分类性能
    """
    def __init__(self, input_dim, num_classes, num_heads=8, dropout=0.1):
        super(AttentionClassifier, self).__init__()
        assert input_dim % num_heads == 0, "input_dim必须能被num_heads整除"
        
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # 自注意力层
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(dropout)
        )
        
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 分类头
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # x: [batch_size, input_dim]
        batch_size = x.size(0)
        
        # 添加序列维度用于注意力计算
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # 自注意力
        residual = x
        x = self.layer_norm1(x)
        Q = self.query(x).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数（这里使用简化的自注意力）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        x = residual + self.dropout(attn_output)
        
        # 前馈网络
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.ffn(x)
        
        # 移除序列维度并分类
        x = x.squeeze(1)  # [batch_size, input_dim]
        logits = self.classifier(x)
        return logits


class ResidualClassifier(nn.Module):
    """
    残差连接分类器
    优点：通过残差连接缓解梯度消失，适合深层网络
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.1):
        super(ResidualClassifier, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.input_proj(x)
        
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # 残差连接
        
        logits = self.classifier(x)
        return logits


class TransformerClassifier(nn.Module):
    """
    Transformer风格分类器
    优点：强大的特征提取能力，适合复杂任务
    """
    def __init__(self, input_dim, num_classes, num_layers=2, num_heads=8, 
                 hidden_dim=None, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 将输入转换为序列格式（添加CLS token）
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.classifier = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, input_dim]
        batch_size = x.size(0)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, 2, input_dim]
        
        # Transformer编码
        x = self.transformer(x)
        
        # 使用CLS token进行分类
        cls_output = x[:, 0, :]  # [batch_size, input_dim]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


class LabelAwareClassifier(nn.Module):
    """
    标签感知分类器（Label-Aware）
    优点：通过标签嵌入增强分类性能，特别适合多分类任务
    """
    def __init__(self, input_dim, num_classes, label_embed_dim=128, dropout=0.1):
        super(LabelAwareClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        
        # 标签嵌入
        self.label_embeddings = nn.Embedding(num_classes, label_embed_dim)
        
        # 特征投影
        self.feature_proj = nn.Linear(input_dim, label_embed_dim)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=label_embed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 分类器
        self.classifier = nn.Linear(label_embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, input_dim]
        batch_size = x.size(0)
        
        # 投影特征
        feature_proj = self.feature_proj(x)  # [batch_size, label_embed_dim]
        feature_proj = feature_proj.unsqueeze(1)  # [batch_size, 1, label_embed_dim]
        
        # 获取所有标签嵌入
        label_ids = torch.arange(self.num_classes, device=x.device)
        label_embeds = self.label_embeddings(label_ids)  # [num_classes, label_embed_dim]
        label_embeds = label_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_classes, label_embed_dim]
        
        # 注意力机制：特征关注标签
        attn_output, _ = self.attention(
            query=feature_proj,
            key=label_embeds,
            value=label_embeds
        )  # [batch_size, 1, label_embed_dim]
        
        attn_output = attn_output.squeeze(1)  # [batch_size, label_embed_dim]
        attn_output = self.dropout(attn_output)
        
        # 分类
        logits = self.classifier(attn_output)  # [batch_size, num_classes]
        return logits


class ContrastiveClassifier(nn.Module):
    """
    对比学习增强的分类器
    优点：通过对比学习提升特征表示质量
    """
    def __init__(self, input_dim, num_classes, projection_dim=256, dropout=0.1):
        super(ContrastiveClassifier, self).__init__()
        
        # 投影层（用于对比学习）
        self.projector = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )
        
    def forward(self, x, return_projection=False):
        # 分类
        logits = self.classifier(x)
        
        if return_projection:
            # 投影（用于对比学习）
            projection = self.projector(x)
            return logits, projection
        return logits


class HierarchicalClassifier(nn.Module):
    """
    层次化分类器
    优点：适合有层次结构的分类任务（如CWE分类）
    """
    def __init__(self, input_dim, num_classes, num_hierarchies=2, dropout=0.1):
        super(HierarchicalClassifier, self).__init__()
        
        self.num_hierarchies = num_hierarchies
        self.hierarchy_classifiers = nn.ModuleList()
        
        # 每个层次一个分类器
        for i in range(num_hierarchies):
            if i == 0:
                dim = input_dim
            else:
                dim = input_dim // (2 ** i)
            
            classifier = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 2, num_classes)
            )
            self.hierarchy_classifiers.append(classifier)
        
        # 融合层
        self.fusion = nn.Linear(num_classes * num_hierarchies, num_classes)
        
    def forward(self, x):
        hierarchy_outputs = []
        
        for i, classifier in enumerate(self.hierarchy_classifiers):
            if i > 0:
                # 降维
                x = F.adaptive_avg_pool1d(x.unsqueeze(1), x.size(-1) // (2 ** i)).squeeze(1)
            output = classifier(x)
            hierarchy_outputs.append(output)
        
        # 融合所有层次
        fused = torch.cat(hierarchy_outputs, dim=-1)
        logits = self.fusion(fused)
        return logits


class CapsuleClassifier(nn.Module):
    """
    胶囊网络分类器
    优点：能够捕获特征之间的空间关系，适合复杂模式识别
    """
    def __init__(self, input_dim, num_classes, num_capsules=10, capsule_dim=16, dropout=0.1):
        super(CapsuleClassifier, self).__init__()
        
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
        # 主胶囊层
        self.primary_capsules = nn.Sequential(
            nn.Linear(input_dim, num_capsules * capsule_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 路由层（简化版）
        self.routing = nn.Sequential(
            nn.Linear(num_capsules * capsule_dim, num_capsules * capsule_dim),
            nn.ReLU()
        )
        
        # 分类胶囊
        self.class_capsules = nn.Linear(num_capsules * capsule_dim, num_classes)
        
    def squash(self, tensor, dim=-1):
        """胶囊网络的squash激活函数"""
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)
        
    def forward(self, x):
        # 主胶囊
        primary = self.primary_capsules(x)  # [batch_size, num_capsules * capsule_dim]
        primary = primary.view(-1, self.num_capsules, self.capsule_dim)
        primary = self.squash(primary, dim=-1)
        
        # 路由
        routed = primary.view(-1, self.num_capsules * self.capsule_dim)
        routed = self.routing(routed)
        
        # 分类
        logits = self.class_capsules(routed)
        return logits

