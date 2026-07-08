"""
风险预测器
加载训练好的模型进行风险预测

说明：模型定义已统一到 smart_elderly_care_v2/models/fusion_net.py，
本文件不再重复定义 MultiModalFusionNet，避免双份代码不一致导致权重加载失败。
"""

import os
import torch
import numpy as np
from typing import Dict, Optional, Tuple

import sys

# 把项目根目录与 V2 目录加入搜索路径，使模型定义成为唯一真源（单一来源 Single Source of Truth）
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # smart_elderly_care_app/
_PROJECT_ROOT = os.path.dirname(_APP_ROOT)                                          # 多模态/
_V2_ROOT = os.path.join(_PROJECT_ROOT, "smart_elderly_care_v2")                     # smart_elderly_care_v2/
for _p in (_PROJECT_ROOT, _V2_ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from config import MODEL_CONFIG, RISK_LEVELS, MODEL_PATH
from inference.missing_handler import MissingDataHandler

# 统一从 V2 导入模型定义（与训练时完全一致）
from models.fusion_net import MultiModalFusionNet


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
            预测结果字典，包含 attention_weights 和 modality_weights（中间结果）
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
        
        # 推理（请求注意力权重用于可解释性展示）
        with torch.no_grad():
            logits, attn_weights_list = self.model(
                video_tensor, audio_tensor, health_tensor, med_tensor,
                return_attention=True,
            )
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1)

        # 提取可学习的模态全局权重（softmax 后，detach 避免 requires_grad 报错）
        modality_names = ["视频", "音频", "生理", "用药"]
        modality_w = torch.softmax(self.model.modality_weights, dim=0).detach().cpu().numpy()

        # 注意力权重：每层 [1, num_heads, 4, 4] → 对 batch+heads 求平均 → [4,4]
        import numpy as _np
        attn_avg = None
        if attn_weights_list:
            mats = []
            for w in attn_weights_list:
                arr = w[0].detach().cpu().numpy()  # [num_heads, 4, 4]
                mats.append(arr.mean(axis=0))      # [4, 4] 对 heads 平均
            attn_avg = _np.stack(mats).mean(axis=0)  # [4, 4] 对 layers 平均
        
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
            # ---- 可解释性中间结果 ----
            'modality_weights': {
                name: float(modality_w[i]) for i, name in enumerate(modality_names)
            },
            'attention_matrix': attn_avg.tolist() if attn_avg is not None else None,
            'modality_names': modality_names,
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
