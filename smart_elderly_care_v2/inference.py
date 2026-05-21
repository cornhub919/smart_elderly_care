"""
推理脚本
使用训练好的模型进行风险预测
"""

import os
import sys
import argparse

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models.fusion_net import MultiModalFusionNet, MultiModalFusionNetLite


class RiskPredictor:
    """风险预测器"""
    
    def __init__(self, checkpoint_path, model_type='full', device=None):
        """
        Args:
            checkpoint_path: 模型检查点路径
            model_type: 模型类型 ('full' 或 'lite')
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = get_config()
        
        # 加载模型
        self._load_model(checkpoint_path, model_type)
        
        # 风险等级定义
        self.risk_levels = {
            0: {"name": "low", "name_cn": "低风险", "description": "记录到系统，写入周报"},
            1: {"name": "medium", "name_cn": "中风险", "description": "推送子女，建议关注"},
            2: {"name": "high", "name_cn": "高风险", "description": "立即报警，通知子女"},
        }
    
    def _load_model(self, checkpoint_path, model_type):
        """加载模型"""
        model_config = self.config['model']
        
        if model_type == 'full':
            self.model = MultiModalFusionNet(model_config)
        else:
            self.model = MultiModalFusionNetLite(model_config)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"模型已加载: {checkpoint_path}")
        if 'val_acc' in checkpoint:
            print(f"模型验证准确率: {checkpoint['val_acc']:.2f}%")
    
    def predict(self, video_features, audio_features, health_features, medication_features):
        """
        预测风险等级
        
        Args:
            video_features: 视频特征向量 [768] 或 [batch, 768]
            audio_features: 音频特征向量 [768] 或 [batch, 768]
            health_features: 生理特征向量 [256] 或 [batch, 256]
            medication_features: 用药特征向量 [128] 或 [batch, 128]
            
        Returns:
            dict: 预测结果
        """
        # 转换为张量
        if not isinstance(video_features, torch.Tensor):
            video_features = torch.tensor(video_features, dtype=torch.float32)
        if not isinstance(audio_features, torch.Tensor):
            audio_features = torch.tensor(audio_features, dtype=torch.float32)
        if not isinstance(health_features, torch.Tensor):
            health_features = torch.tensor(health_features, dtype=torch.float32)
        if not isinstance(medication_features, torch.Tensor):
            medication_features = torch.tensor(medication_features, dtype=torch.float32)
        
        # 添加批次维度
        if video_features.dim() == 1:
            video_features = video_features.unsqueeze(0)
            audio_features = audio_features.unsqueeze(0)
            health_features = health_features.unsqueeze(0)
            medication_features = medication_features.unsqueeze(0)
        
        # 移动到设备
        video_features = video_features.to(self.device)
        audio_features = audio_features.to(self.device)
        health_features = health_features.to(self.device)
        medication_features = medication_features.to(self.device)
        
        # 推理
        with torch.no_grad():
            logits = self.model(video_features, audio_features, health_features, medication_features)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        # 构建结果
        results = []
        for i in range(len(predictions)):
            pred_class = predictions[i].item()
            pred_prob = probs[i].cpu().numpy()
            
            result = {
                'risk_level': pred_class,
                'risk_name': self.risk_levels[pred_class]['name'],
                'risk_name_cn': self.risk_levels[pred_class]['name_cn'],
                'description': self.risk_levels[pred_class]['description'],
                'confidence': pred_prob[pred_class],
                'probabilities': {
                    'low': float(pred_prob[0]),
                    'medium': float(pred_prob[1]),
                    'high': float(pred_prob[2]),
                }
            }
            results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    def predict_from_file(self, data_path):
        """从文件加载特征并预测"""
        video_features = np.load(os.path.join(data_path, 'video_features.npy'))
        audio_features = np.load(os.path.join(data_path, 'audio_features.npy'))
        health_features = np.load(os.path.join(data_path, 'health_features.npy'))
        medication_features = np.load(os.path.join(data_path, 'medication_features.npy'))
        
        return self.predict(video_features, audio_features, health_features, medication_features)


def demo():
    """演示推理过程"""
    print("="*60)
    print("智护家 V2 - 多模态风险预测演示")
    print("="*60)
    
    # 检查模型是否存在
    config = get_config()
    checkpoint_path = os.path.join(config['data']['checkpoint_dir'], 'best_model.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"\n错误: 模型文件不存在: {checkpoint_path}")
        print("请先运行 train.py 训练模型")
        return
    
    # 创建预测器
    predictor = RiskPredictor(checkpoint_path)
    
    # 模拟输入数据
    print("\n生成模拟输入数据...")
    batch_size = 5
    video_features = torch.randn(batch_size, 768)
    audio_features = torch.randn(batch_size, 768)
    health_features = torch.randn(batch_size, 256)
    medication_features = torch.randn(batch_size, 128)
    
    # 预测
    print("\n进行风险预测...")
    results = predictor.predict(video_features, audio_features, health_features, medication_features)
    
    # 显示结果
    print("\n预测结果:")
    print("-"*60)
    for i, result in enumerate(results):
        print(f"\n样本 {i+1}:")
        print(f"  风险等级: {result['risk_name_cn']} ({result['risk_name']})")
        print(f"  置信度: {result['confidence']:.2%}")
        print(f"  概率分布: 低风险={result['probabilities']['low']:.2%}, "
              f"中风险={result['probabilities']['medium']:.2%}, "
              f"高风险={result['probabilities']['high']:.2%}")
        print(f"  处理建议: {result['description']}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型进行风险预测')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--model', type=str, default='full', choices=['full', 'lite'], help='模型类型')
    parser.add_argument('--demo', action='store_true', help='运行演示模式')
    
    # 输入数据路径
    parser.add_argument('--video', type=str, default=None, help='视频特征文件路径')
    parser.add_argument('--audio', type=str, default=None, help='音频特征文件路径')
    parser.add_argument('--health', type=str, default=None, help='生理特征文件路径')
    parser.add_argument('--medication', type=str, default=None, help='用药特征文件路径')
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
        return
    
    # 检查输入
    if not all([args.video, args.audio, args.health, args.medication]):
        print("错误: 请提供所有模态的特征文件路径，或使用 --demo 运行演示模式")
        return
    
    # 确定检查点路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        config = get_config()
        checkpoint_path = os.path.join(config['data']['checkpoint_dir'], 'best_model.pt')
    
    # 创建预测器
    predictor = RiskPredictor(checkpoint_path, args.model)
    
    # 加载特征
    video_features = np.load(args.video)
    audio_features = np.load(args.audio)
    health_features = np.load(args.health)
    medication_features = np.load(args.medication)
    
    # 预测
    results = predictor.predict(video_features, audio_features, health_features, medication_features)
    
    # 显示结果
    if isinstance(results, dict):
        results = [results]
    
    for i, result in enumerate(results):
        print(f"\n样本 {i+1}:")
        print(f"  风险等级: {result['risk_name_cn']}")
        print(f"  置信度: {result['confidence']:.2%}")
        print(f"  处理建议: {result['description']}")


if __name__ == "__main__":
    main()
