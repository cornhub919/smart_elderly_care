"""
风险预测器
加载训练好的模型进行风险预测
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import math

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CONFIG, RISK_LEVELS, MODEL_PATH
from inference.missing_handler import MissingDataHandler


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.out_proj(output)
        
        return output, attn_weights


class MultiModalFusionNet(nn.Module):
    """多模态融合网络（与训练时保持一致）"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        video_dim = config['video_encoder']['hidden_dim']
        audio_dim = config['audio_encoder']['hidden_dim']
        health_dim = config['health_encoder']['hidden_dim']
        med_dim = config['medication_encoder']['embedding_dim']
        fusion_dim = config['fusion']['hidden_dim']
        num_heads = config['fusion']['num_heads']
        num_layers = config['fusion']['num_layers']
        dropout = config['fusion']['dropout']
        num_classes = config['classifier']['num_classes']
        
        # 特征投影层
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
        classifier_hidden = config['classifier']['hidden_dims']
        classifier_dropout = config['classifier']['dropout']
        
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
        
        # 模态权重
        self.modality_weights = nn.Parameter(torch.ones(4) / 4)
        
    def forward(self, video_feat, audio_feat, health_feat, med_feat):
        batch_size = video_feat.size(0)
        
        # 投影
        v = self.video_proj(video_feat)
        a = self.audio_proj(audio_feat)
        h = self.health_proj(health_feat)
        m = self.med_proj(med_feat)
        
        # 模态嵌入
        modality_ids = torch.arange(4, device=video_feat.device)
        modality_embeds = self.modality_embedding(modality_ids)
        
        v = v + modality_embeds[0]
        a = a + modality_embeds[1]
        h = h + modality_embeds[2]
        m = m + modality_embeds[3]
        
        # 堆叠
        features = torch.stack([v, a, h, m], dim=1)
        
        # 模态权重
        weights = torch.softmax(self.modality_weights, dim=0)
        features = features * weights.view(1, 4, 1)
        
        # Cross-Modal Attention
        for cross_attn in self.cross_attention_layers:
            features_attended, _ = cross_attn(features, features, features)
            features = features + features_attended
        
        # 自注意力
        features_normed = self.norm1(features)
        self_attn_output, _ = self.self_attention(
            features_normed, features_normed, features_normed
        )
        features = features + self_attn_output
        
        # 前馈网络
        features = features + self.ffn(self.norm2(features))
        
        # 全局池化
        fused_features = features.mean(dim=1)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits


class RiskPredictor:
    """风险预测器"""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径，默认使用配置中的路径
            device: 计算设备
        """
        self.model_path = model_path or MODEL_PATH
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self._load_model()
        
        # 缺失数据处理器
        self.missing_handler = MissingDataHandler()
        
        # 风险等级定义
        self.risk_levels = RISK_LEVELS
    
    def _load_model(self) -> MultiModalFusionNet:
        """加载模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 创建模型
        model = MultiModalFusionNet(MODEL_CONFIG)
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device,weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型加载成功，验证准确率: {checkpoint.get('val_acc', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, video_features: np.ndarray = None,
                audio_features: np.ndarray = None,
                health_features: np.ndarray = None,
                medication_features: np.ndarray = None) -> Dict:
        """
        进行风险预测
        
        Args:
            video_features: 视频特征 [768] 或 None
            audio_features: 音频特征 [768] 或 None
            health_features: 生理特征 [256] 或 None
            medication_features: 用药特征 [128] 或 None
            
        Returns:
            预测结果字典
        """
        # 处理缺失数据
        features_dict = {
            'video': video_features,
            'audio': audio_features,
            'health': health_features,
            'medication': medication_features
        }
        
        processed_features = self.missing_handler.handle_batch(features_dict)
        missing_info = self.missing_handler.get_missing_info(features_dict)
        
        # 转换为张量
        video_tensor = torch.tensor(processed_features['video'], dtype=torch.float32).unsqueeze(0)
        audio_tensor = torch.tensor(processed_features['audio'], dtype=torch.float32).unsqueeze(0)
        health_tensor = torch.tensor(processed_features['health'], dtype=torch.float32).unsqueeze(0)
        med_tensor = torch.tensor(processed_features['medication'], dtype=torch.float32).unsqueeze(0)
        
        # 移动到设备
        video_tensor = video_tensor.to(self.device)
        audio_tensor = audio_tensor.to(self.device)
        health_tensor = health_tensor.to(self.device)
        med_tensor = med_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            logits = self.model(video_tensor, audio_tensor, health_tensor, med_tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1)
        
        # 构建结果
        pred_class = prediction.item()
        pred_probs = probs[0].cpu().numpy()
        
        result = {
            'risk_level': pred_class,
            'risk_name': self.risk_levels[pred_class]['name'],
            'risk_name_cn': self.risk_levels[pred_class]['name_cn'],
            'confidence': float(pred_probs[pred_class]),
            'probabilities': {
                'low': float(pred_probs[0]),
                'medium': float(pred_probs[1]),
                'high': float(pred_probs[2]),
            },
            'description': self.risk_levels[pred_class]['description'],
            'action': self.risk_levels[pred_class]['action'],
            'color': self.risk_levels[pred_class]['color'],
            'missing_modalities': [k for k, v in missing_info.items() if v],
        }
        
        return result
    
    def predict_batch(self, features_batch: Dict[str, np.ndarray]) -> list:
        """
        批量预测
        
        Args:
            features_batch: 各模态特征字典，值为 [batch, dim] 数组
            
        Returns:
            预测结果列表
        """
        results = []
        
        # 获取批次大小
        batch_size = 0
        for key, value in features_batch.items():
            if value is not None:
                batch_size = len(value)
                break
        
        if batch_size == 0:
            return results
        
        # 逐样本预测
        for i in range(batch_size):
            sample_features = {
                key: value[i] if value is not None else None
                for key, value in features_batch.items()
            }
            
            result = self.predict(
                video_features=sample_features.get('video'),
                audio_features=sample_features.get('audio'),
                health_features=sample_features.get('health'),
                medication_features=sample_features.get('medication')
            )
            results.append(result)
        
        return results


def demo():
    """演示预测功能"""
    print("="*60)
    print("智护家 - 风险预测演示")
    print("="*60)
    
    # 创建预测器
    try:
        predictor = RiskPredictor()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保模型文件存在于 pretrained_models/fusion_model.pt")
        return
    
    # 测试1：完整数据
    print("\n测试1: 完整数据")
    video_feat = np.random.randn(768).astype(np.float32)
    audio_feat = np.random.randn(768).astype(np.float32)
    health_feat = np.random.randn(256).astype(np.float32)
    med_feat = np.random.randn(128).astype(np.float32)
    
    result = predictor.predict(video_feat, audio_feat, health_feat, med_feat)
    print(f"  风险等级: {result['risk_name_cn']}")
    print(f"  置信度: {result['confidence']:.2%}")
    print(f"  概率分布: 低={result['probabilities']['low']:.2%}, "
          f"中={result['probabilities']['medium']:.2%}, "
          f"高={result['probabilities']['high']:.2%}")
    
    # 测试2：缺失数据
    print("\n测试2: 缺失视频和音频数据")
    result = predictor.predict(
        video_features=None,
        audio_features=None,
        health_features=health_feat,
        medication_features=med_feat
    )
    print(f"  风险等级: {result['risk_name_cn']}")
    print(f"  缺失模态: {result['missing_modalities']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    demo()
