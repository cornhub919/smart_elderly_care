"""
数据准备脚本
生成模拟多模态数据集用于训练
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from tqdm import tqdm
import json
import cv2
from PIL import Image

from config import get_config


class MultiModalDataGenerator:
    """多模态数据生成器"""
    
    def __init__(self, config=None, seed=42):
        self.config = config or get_config()
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 风险场景定义
        self.risk_scenarios = {
            0: self._generate_low_risk,      # 低风险
            1: self._generate_medium_risk,   # 中风险
            2: self._generate_high_risk,     # 高风险
        }
    
    def generate_dataset(self, num_samples=1000, save_path=None):
        """
        生成完整的多模态数据集
        
        Args:
            num_samples: 样本数量
            save_path: 保存路径
            
        Returns:
            dict: 包含所有模态数据和标签的字典
        """
        print(f"生成 {num_samples} 个多模态样本...")
        
        # 确定每个风险等级的样本数（模拟真实分布）
        # 低风险 70%，中风险 20%，高风险 10%
        risk_distribution = {
            0: int(num_samples * 0.7),
            1: int(num_samples * 0.2),
            2: num_samples - int(num_samples * 0.7) - int(num_samples * 0.2)
        }
        
        dataset = {
            'video_features': [],
            'audio_features': [],
            'health_features': [],
            'medication_features': [],
            'labels': [],
            'timestamps': [],
            'metadata': []
        }
        
        for risk_level, count in risk_distribution.items():
            print(f"  生成风险等级 {risk_level} ({self.config.risk_levels[risk_level].name_cn}): {count} 个样本")
            
            for _ in tqdm(range(count), desc=f"Risk {risk_level}"):
                sample = self.risk_scenarios[risk_level]()
                
                dataset['video_features'].append(sample['video_features'])
                dataset['audio_features'].append(sample['audio_features'])
                dataset['health_features'].append(sample['health_features'])
                dataset['medication_features'].append(sample['medication_features'])
                dataset['labels'].append(risk_level)
                dataset['timestamps'].append(datetime.now().isoformat())
                dataset['metadata'].append(sample['metadata'])
        
        # 转换为numpy数组
        for key in ['video_features', 'audio_features', 'health_features', 'medication_features']:
            dataset[key] = np.array(dataset[key], dtype=np.float32)
        dataset['labels'] = np.array(dataset['labels'], dtype=np.int64)
        
        # 打乱数据
        indices = np.random.permutation(len(dataset['labels']))
        for key in ['video_features', 'audio_features', 'health_features', 'medication_features', 'labels']:
            dataset[key] = dataset[key][indices]
        dataset['timestamps'] = [dataset['timestamps'][i] for i in indices]
        dataset['metadata'] = [dataset['metadata'][i] for i in indices]
        
        # 保存数据集
        if save_path:
            self._save_dataset(dataset, save_path)
        
        print(f"数据集生成完成！")
        print(f"  视频特征: {dataset['video_features'].shape}")
        print(f"  音频特征: {dataset['audio_features'].shape}")
        print(f"  生理特征: {dataset['health_features'].shape}")
        print(f"  用药特征: {dataset['medication_features'].shape}")
        print(f"  标签分布: {np.bincount(dataset['labels'])}")
        
        return dataset
    
    def _generate_low_risk(self):
        """生成低风险样本"""
        # 视频特征：正常活动，姿态稳定
        video_features = self._generate_video_features(
            motion_level='normal',
            fall_probability=0.0
        )
        
        # 音频特征：正常环境音
        audio_features = self._generate_audio_features(
            has_impact=False,
            has_help_call=False,
            noise_level='low'
        )
        
        # 生理特征：正常范围
        health_features = self._generate_health_features(
            heart_rate_range=(60, 100),
            blood_oxygen_range=(95, 100),
            blood_pressure_range=(90, 140, 60, 90),
            activity_level='normal'
        )
        
        # 用药特征：按时服药
        medication_features = self._generate_medication_features(
            adherence_rate=0.95,
            missed_doses=0
        )
        
        return {
            'video_features': video_features,
            'audio_features': audio_features,
            'health_features': health_features,
            'medication_features': medication_features,
            'metadata': {'scenario': 'normal_activity'}
        }
    
    def _generate_medium_risk(self):
        """生成中风险样本"""
        scenario = np.random.choice(['abnormal_behavior', 'health_warning', 'medication_issue'])
        
        if scenario == 'abnormal_behavior':
            # 视频特征：异常行为（长时间静止、夜间活动）
            video_features = self._generate_video_features(
                motion_level='low',
                fall_probability=0.0,
                stillness_duration=np.random.uniform(30, 60)
            )
            audio_features = self._generate_audio_features(has_impact=False, has_help_call=False)
            health_features = self._generate_health_features()
            medication_features = self._generate_medication_features()
            
        elif scenario == 'health_warning':
            # 生理特征：轻微异常
            video_features = self._generate_video_features()
            audio_features = self._generate_audio_features()
            health_features = self._generate_health_features(
                heart_rate_range=(100, 120),  # 心率偏高
                blood_oxygen_range=(90, 95),  # 血氧偏低
                blood_pressure_range=(140, 160, 90, 100),  # 血压偏高
                activity_level='low'
            )
            medication_features = self._generate_medication_features()
            
        else:  # medication_issue
            video_features = self._generate_video_features()
            audio_features = self._generate_audio_features()
            health_features = self._generate_health_features()
            medication_features = self._generate_medication_features(
                adherence_rate=0.6,
                missed_doses=np.random.randint(1, 3)
            )
        
        return {
            'video_features': video_features,
            'audio_features': audio_features,
            'health_features': health_features,
            'medication_features': medication_features,
            'metadata': {'scenario': scenario}
        }
    
    def _generate_high_risk(self):
        """生成高风险样本"""
        scenario = np.random.choice(['fall', 'health_crisis', 'help_call'])
        
        if scenario == 'fall':
            # 视频特征：跌倒
            video_features = self._generate_video_features(
                motion_level='sudden_drop',
                fall_probability=1.0,
                stillness_duration=np.random.uniform(60, 180)
            )
            # 音频特征：撞击声
            audio_features = self._generate_audio_features(
                has_impact=True,
                has_help_call=np.random.random() > 0.5
            )
            # 生理特征：心率异常升高
            health_features = self._generate_health_features(
                heart_rate_range=(120, 150),
                blood_oxygen_range=(85, 92),
                activity_level='sudden_change'
            )
            medication_features = self._generate_medication_features()
            
        elif scenario == 'health_crisis':
            video_features = self._generate_video_features(motion_level='low')
            audio_features = self._generate_audio_features(has_help_call=np.random.random() > 0.5)
            health_features = self._generate_health_features(
                heart_rate_range=(130, 160),
                blood_oxygen_range=(80, 90),
                blood_pressure_range=(160, 200, 100, 120),
                activity_level='crisis'
            )
            medication_features = self._generate_medication_features()
            
        else:  # help_call
            video_features = self._generate_video_features(motion_level='low')
            audio_features = self._generate_audio_features(
                has_impact=np.random.random() > 0.5,
                has_help_call=True
            )
            health_features = self._generate_health_features(
                heart_rate_range=(100, 140),
                blood_oxygen_range=(88, 95)
            )
            medication_features = self._generate_medication_features()
        
        return {
            'video_features': video_features,
            'audio_features': audio_features,
            'health_features': health_features,
            'medication_features': medication_features,
            'metadata': {'scenario': scenario}
        }
    
    def _generate_video_features(self, motion_level='normal', fall_probability=0.0, 
                                  stillness_duration=0):
        """生成视频特征向量"""
        hidden_dim = self.config.model.video_encoder.hidden_dim  # 768
        
        # 基础特征（模拟 VideoMAE 输出）
        base_features = np.random.randn(hidden_dim).astype(np.float32) * 0.1
        
        if motion_level == 'normal':
            # 正常活动：特征分布均匀
            base_features += np.random.randn(hidden_dim).astype(np.float32) * 0.3
        elif motion_level == 'low':
            # 低活动：特征偏向静止模式
            base_features[:256] += 0.5  # 静止特征增强
        elif motion_level == 'sudden_drop':
            # 跌倒：特征有明显的跌倒模式
            base_features[256:512] += 1.0  # 跌倒特征增强
            base_features[512:768] += stillness_duration / 100  # 静止时长编码
        
        if fall_probability > 0.5:
            # 跌倒特征
            base_features[200:300] += np.random.randn(100).astype(np.float32) * 0.8
        
        return base_features
    
    def _generate_audio_features(self, has_impact=False, has_help_call=False, 
                                  noise_level='normal'):
        """生成音频特征向量"""
        hidden_dim = self.config.model.audio_encoder.hidden_dim  # 768
        
        base_features = np.random.randn(hidden_dim).astype(np.float32) * 0.1
        
        if has_impact:
            # 撞击声特征
            base_features[0:128] += np.random.randn(128).astype(np.float32) * 1.0
        
        if has_help_call:
            # 呼救声特征
            base_features[128:256] += np.random.randn(128).astype(np.float32) * 1.2
        
        if noise_level == 'high':
            base_features += np.random.randn(hidden_dim).astype(np.float32) * 0.5
        
        return base_features
    
    def _generate_health_features(self, heart_rate_range=(60, 100), 
                                   blood_oxygen_range=(95, 100),
                                   blood_pressure_range=(90, 140, 60, 90),
                                   activity_level='normal'):
        """生成生理数据特征"""
        seq_len = self.config.data.health.sequence_length  # 100
        num_features = len(self.config.data.health.features)  # 5
        
        features = np.zeros((seq_len, num_features), dtype=np.float32)
        
        # 心率
        features[:, 0] = np.random.uniform(*heart_rate_range, seq_len)
        features[:, 0] += np.sin(np.linspace(0, 4*np.pi, seq_len)) * 5  # 添加波动
        
        # 血氧
        features[:, 1] = np.random.uniform(*blood_oxygen_range, seq_len)
        
        # 收缩压
        features[:, 2] = np.random.uniform(blood_pressure_range[0], blood_pressure_range[1], seq_len)
        
        # 舒张压
        features[:, 3] = np.random.uniform(blood_pressure_range[2], blood_pressure_range[3], seq_len)
        
        # 步数
        if activity_level == 'normal':
            features[:, 4] = np.random.exponential(100, seq_len)
        elif activity_level == 'low':
            features[:, 4] = np.random.exponential(20, seq_len)
        elif activity_level == 'crisis':
            features[:, 4] = np.zeros(seq_len)
        elif activity_level == 'sudden_change':
            features[:50, 4] = np.random.exponential(100, 50)
            features[50:, 4] = 0  # 突然停止
        
        # 展平为特征向量
        # 这里我们用一个简单的 Transformer 编码器输出
        hidden_dim = self.config.model.health_encoder.hidden_dim  # 256
        flattened = features.flatten()
        projected = np.random.randn(hidden_dim).astype(np.float32) * 0.1
        projected[:min(len(flattened), hidden_dim)] += flattened[:hidden_dim] * 0.01
        
        return projected
    
    def _generate_medication_features(self, adherence_rate=0.9, missed_doses=0):
        """生成用药特征"""
        embed_dim = self.config.data.medication.embedding_dim  # 128
        
        features = np.random.randn(embed_dim).astype(np.float32) * 0.1
        
        # 依从率编码
        features[0:32] += adherence_rate
        
        # 漏服次数编码
        features[32:64] += missed_doses * 0.3
        
        return features
    
    def _save_dataset(self, dataset, save_path):
        """保存数据集"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存特征数据
        np.save(os.path.join(save_path, 'video_features.npy'), dataset['video_features'])
        np.save(os.path.join(save_path, 'audio_features.npy'), dataset['audio_features'])
        np.save(os.path.join(save_path, 'health_features.npy'), dataset['health_features'])
        np.save(os.path.join(save_path, 'medication_features.npy'), dataset['medication_features'])
        np.save(os.path.join(save_path, 'labels.npy'), dataset['labels'])
        
        # 保存元数据
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump({
                'timestamps': dataset['timestamps'],
                'metadata': dataset['metadata'],
                'num_samples': len(dataset['labels']),
                'label_distribution': {str(k): int(v) for k, v in enumerate(np.bincount(dataset['labels']))}
            }, f, indent=2)
        
        print(f"数据集已保存到: {save_path}")


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    """划分训练集、验证集、测试集"""
    num_samples = len(dataset['labels'])
    indices = np.random.permutation(num_samples)
    
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    def get_subset(indices):
        return {
            'video_features': dataset['video_features'][indices],
            'audio_features': dataset['audio_features'][indices],
            'health_features': dataset['health_features'][indices],
            'medication_features': dataset['medication_features'][indices],
            'labels': dataset['labels'][indices],
        }
    
    return {
        'train': get_subset(train_indices),
        'val': get_subset(val_indices),
        'test': get_subset(test_indices)
    }


if __name__ == "__main__":
    # 生成数据集
    config = get_config()
    generator = MultiModalDataGenerator(config)
    
    # 生成训练数据
    dataset = generator.generate_dataset(
        num_samples=5000,
        save_path=config.data.processed_data_dir
    )
    
    # 划分数据集
    splits = split_dataset(dataset)
    
    # 保存划分后的数据
    for split_name, split_data in splits.items():
        split_path = os.path.join(config.data.processed_data_dir, split_name)
        os.makedirs(split_path, exist_ok=True)
        
        for key, value in split_data.items():
            np.save(os.path.join(split_path, f'{key}.npy'), value)
        
        print(f"{split_name}集: {len(split_data['labels'])} 个样本")
