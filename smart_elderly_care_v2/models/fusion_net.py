"""
多模态融合网络
实现 Cross-Attention 融合机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制
    让不同模态之间相互交互
    """
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value):
        """
        Args:
            query: [batch, seq_len_q, hidden_dim]
            key: [batch, seq_len_k, hidden_dim]
            value: [batch, seq_len_v, hidden_dim]
        Returns:
            output: [batch, seq_len_q, hidden_dim]
            attention_weights: [batch, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # 投影
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        output = torch.matmul(attn_weights, V)
        
        # 重塑回来
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.out_proj(output)
        
        return output, attn_weights


class MultiModalFusionNet(nn.Module):
    """
    多模态融合网络
    
    架构：
    1. 各模态特征投影到统一维度
    2. Cross-Modal Attention 融合
    3. 分类头输出风险等级
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 获取配置
        video_dim = config['video_encoder']['hidden_dim']      # 768
        audio_dim = config['audio_encoder']['hidden_dim']      # 768
        health_dim = config['health_encoder']['hidden_dim']    # 256
        med_dim = config['medication_encoder']['embedding_dim'] # 128
        fusion_dim = config['fusion']['hidden_dim']            # 512
        num_heads = config['fusion']['num_heads']              # 8
        num_layers = config['fusion']['num_layers']            # 4
        dropout = config['fusion']['dropout']                  # 0.2
        num_classes = config['classifier']['num_classes']      # 3
        
        # 特征投影层（对齐维度）
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.health_proj = nn.Sequential(
            nn.Linear(health_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.med_proj = nn.Sequential(
            nn.Linear(med_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 模态类型嵌入
        self.modality_embedding = nn.Embedding(4, fusion_dim)
        
        # Cross-Modal Attention 层
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttention(fusion_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        
        # 分类头
        classifier_hidden = config['classifier']['hidden_dims']  # [256, 128]
        classifier_dropout = config['classifier']['dropout']     # 0.3
        
        classifier_layers = []
        input_dim = fusion_dim
        for hidden_dim in classifier_hidden:
            classifier_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(classifier_dropout)
            ])
            input_dim = hidden_dim
        
        classifier_layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        
        # 模态权重（可学习）
        self.modality_weights = nn.Parameter(torch.ones(4) / 4)
        
    def forward(self, video_feat, audio_feat, health_feat, med_feat, return_attention=False):
        """
        Args:
            video_feat: [batch, video_dim] 视频特征
            audio_feat: [batch, audio_dim] 音频特征
            health_feat: [batch, health_dim] 生理特征
            med_feat: [batch, med_dim] 用药特征
            return_attention: 是否返回注意力权重
            
        Returns:
            logits: [batch, num_classes] 分类 logits
            attention_weights: (可选) 注意力权重
        """
        batch_size = video_feat.size(0)
        
        # 投影到统一维度
        v = self.video_proj(video_feat)      # [batch, fusion_dim]
        a = self.audio_proj(audio_feat)      # [batch, fusion_dim]
        h = self.health_proj(health_feat)    # [batch, fusion_dim]
        m = self.med_proj(med_feat)          # [batch, fusion_dim]
        
        # 添加模态类型嵌入
        modality_ids = torch.arange(4, device=video_feat.device)
        modality_embeds = self.modality_embedding(modality_ids)  # [4, fusion_dim]
        
        v = v + modality_embeds[0]
        a = a + modality_embeds[1]
        h = h + modality_embeds[2]
        m = m + modality_embeds[3]
        
        # 堆叠为序列 [batch, 4, fusion_dim]
        features = torch.stack([v, a, h, m], dim=1)
        
        # 应用可学习的模态权重
        weights = F.softmax(self.modality_weights, dim=0)
        features = features * weights.view(1, 4, 1)
        
        # Cross-Modal Attention
        attention_weights_list = []
        for cross_attn in self.cross_attention_layers:
            features_attended, attn_weights = cross_attn(features, features, features)
            features = features + features_attended
            attention_weights_list.append(attn_weights)
        
        # 自注意力
        features_normed = self.norm1(features)
        self_attn_output, _ = self.self_attention(
            features_normed, features_normed, features_normed
        )
        features = features + self_attn_output
        
        # 前馈网络
        features = features + self.ffn(self.norm2(features))
        
        # 全局池化
        fused_features = features.mean(dim=1)  # [batch, fusion_dim]
        
        # 分类
        logits = self.classifier(fused_features)  # [batch, num_classes]
        
        if return_attention:
            return logits, attention_weights_list
        return logits
    
    def get_features(self, video_feat, audio_feat, health_feat, med_feat):
        """获取融合后的特征向量（不进行分类）"""
        batch_size = video_feat.size(0)
        
        v = self.video_proj(video_feat)
        a = self.audio_proj(audio_feat)
        h = self.health_proj(health_feat)
        m = self.med_proj(med_feat)
        
        features = torch.stack([v, a, h, m], dim=1)
        
        for cross_attn in self.cross_attention_layers:
            features_attended, _ = cross_attn(features, features, features)
            features = features + features_attended
        
        fused_features = features.mean(dim=1)
        return fused_features


class MultiModalFusionNetLite(nn.Module):
    """
    轻量版多模态融合网络
    使用简单的 MLP 融合，适合快速实验
    """
    
    def __init__(self, config):
        super().__init__()
        
        video_dim = config['video_encoder']['hidden_dim']
        audio_dim = config['audio_encoder']['hidden_dim']
        health_dim = config['health_encoder']['hidden_dim']
        med_dim = config['medication_encoder']['embedding_dim']
        fusion_dim = config['fusion']['hidden_dim']
        num_classes = config['classifier']['num_classes']
        
        # 特征投影
        self.video_proj = nn.Linear(video_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.health_proj = nn.Linear(health_dim, fusion_dim)
        self.med_proj = nn.Linear(med_dim, fusion_dim)
        
        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, video_feat, audio_feat, health_feat, med_feat):
        # 投影
        v = self.video_proj(video_feat)
        a = self.audio_proj(audio_feat)
        h = self.health_proj(health_feat)
        m = self.med_proj(med_feat)
        
        # 拼接
        concat = torch.cat([v, a, h, m], dim=-1)
        
        # 融合
        fused = self.fusion(concat)
        
        # 分类
        logits = self.classifier(fused)
        
        return logits


# 测试代码
if __name__ == "__main__":
    # 测试配置
    config = {
        'video_encoder': {'hidden_dim': 768},
        'audio_encoder': {'hidden_dim': 768},
        'health_encoder': {'hidden_dim': 256},
        'medication_encoder': {'embedding_dim': 128},
        'fusion': {'hidden_dim': 512, 'num_heads': 8, 'num_layers': 4, 'dropout': 0.2},
        'classifier': {'num_classes': 3, 'hidden_dims': [256, 128], 'dropout': 0.3}
    }
    
    batch_size = 4
    
    # 创建模型
    model = MultiModalFusionNet(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    video_feat = torch.randn(batch_size, 768)
    audio_feat = torch.randn(batch_size, 768)
    health_feat = torch.randn(batch_size, 256)
    med_feat = torch.randn(batch_size, 128)
    
    logits, attention = model(video_feat, audio_feat, health_feat, med_feat, return_attention=True)
    print(f"输出 logits: {logits.shape}")
    print(f"注意力权重层数: {len(attention)}")
    
    # 测试轻量版
    model_lite = MultiModalFusionNetLite(config)
    print(f"轻量版参数量: {sum(p.numel() for p in model_lite.parameters()):,}")
    
    logits_lite = model_lite(video_feat, audio_feat, health_feat, med_feat)
    print(f"轻量版输出: {logits_lite.shape}")
