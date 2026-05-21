"""
多模态编码器定义
包含视频、音频、生理数据、用药数据的编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VideoEncoder(nn.Module):
    """
    视频编码器
    使用 VideoMAE 或简单的 3D CNN 提取视频特征
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config['hidden_dim']
        
        # 简化版：使用 3D CNN + Transformer
        # 实际部署时可替换为预训练的 VideoMAE
        
        # 3D 卷积层
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 投影层
        self.proj = nn.Linear(256, self.hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, channels, frames, height, width]
        Returns:
            features: [batch, hidden_dim]
        """
        # 3D 卷积
        x = self.conv3d(x)  # [batch, 256, T', H', W']
        
        # 展平空间维度
        batch_size = x.size(0)
        x = x.flatten(2)  # [batch, 256, T*H*W]
        x = x.transpose(1, 2)  # [batch, T*H*W, 256]
        
        # 投影到 hidden_dim
        x = self.proj(x)  # [batch, T*H*W, hidden_dim]
        
        # Transformer 编码
        x = self.transformer(x)  # [batch, T*H*W, hidden_dim]
        
        # 全局平均池化
        features = x.mean(dim=1)  # [batch, hidden_dim]
        
        return features


class AudioEncoder(nn.Module):
    """
    音频编码器
    使用 AST (Audio Spectrogram Transformer) 或 CNN
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config['hidden_dim']
        
        # CNN 特征提取
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 投影层
        self.proj = nn.Linear(256, self.hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, 1, n_mels, time_steps] Mel频谱图
        Returns:
            features: [batch, hidden_dim]
        """
        # CNN 特征提取
        x = self.conv(x)  # [batch, 256, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [batch, 256]
        
        # 投影
        x = self.proj(x)  # [batch, hidden_dim]
        
        # 扩展序列维度用于 Transformer
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        x = self.transformer(x)  # [batch, 1, hidden_dim]
        
        features = x.squeeze(1)  # [batch, hidden_dim]
        
        return features


class HealthEncoder(nn.Module):
    """
    生理数据编码器
    使用 Transformer 处理时间序列数据
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.hidden_dim, dropout=0.1)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config['num_heads'],
            dim_feedforward=512,
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim] 生理数据时间序列
        Returns:
            features: [batch, hidden_dim]
        """
        # 输入嵌入
        x = self.input_embedding(x)  # [batch, seq_len, hidden_dim]
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer 编码
        x = self.transformer(x)  # [batch, seq_len, hidden_dim]
        
        # 全局平均池化
        features = x.mean(dim=1)  # [batch, hidden_dim]
        
        return features


class MedicationEncoder(nn.Module):
    """
    用药数据编码器
    使用嵌入层处理离散用药事件
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.num_medications = config.get('num_medications', 100)
        
        # 用药嵌入
        self.medication_embedding = nn.Embedding(self.num_medications, self.embedding_dim)
        
        # 时间编码
        self.time_embedding = nn.Linear(1, self.embedding_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
    def forward(self, medication_ids, times):
        """
        Args:
            medication_ids: [batch, num_medications] 药品ID
            times: [batch, num_medications, 1] 服药时间
        Returns:
            features: [batch, embedding_dim]
        """
        # 药品嵌入
        med_embed = self.medication_embedding(medication_ids)  # [batch, num_meds, embed_dim]
        
        # 时间嵌入
        time_embed = self.time_embedding(times)  # [batch, num_meds, embed_dim]
        
        # 拼接并融合
        combined = torch.cat([med_embed, time_embed], dim=-1)  # [batch, num_meds, embed_dim*2]
        fused = self.fusion(combined)  # [batch, num_meds, embed_dim]
        
        # 平均池化
        features = fused.mean(dim=1)  # [batch, embed_dim]
        
        return features


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 测试代码
if __name__ == "__main__":
    # 测试各编码器
    batch_size = 4
    
    # 视频编码器
    video_config = {'hidden_dim': 768}
    video_encoder = VideoEncoder(video_config)
    video_input = torch.randn(batch_size, 3, 16, 224, 224)
    video_features = video_encoder(video_input)
    print(f"视频特征: {video_features.shape}")
    
    # 音频编码器
    audio_config = {'hidden_dim': 768}
    audio_encoder = AudioEncoder(audio_config)
    audio_input = torch.randn(batch_size, 1, 128, 100)
    audio_features = audio_encoder(audio_input)
    print(f"音频特征: {audio_features.shape}")
    
    # 生理数据编码器
    health_config = {'input_dim': 5, 'hidden_dim': 256, 'num_heads': 8, 'num_layers': 4, 'dropout': 0.1}
    health_encoder = HealthEncoder(health_config)
    health_input = torch.randn(batch_size, 100, 5)
    health_features = health_encoder(health_input)
    print(f"生理特征: {health_features.shape}")
    
    # 用药编码器
    med_config = {'embedding_dim': 128, 'num_medications': 100}
    med_encoder = MedicationEncoder(med_config)
    med_ids = torch.randint(0, 100, (batch_size, 10))
    med_times = torch.rand(batch_size, 10, 1)
    med_features = med_encoder(med_ids, med_times)
    print(f"用药特征: {med_features.shape}")
