"""
数据加载器
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple


class MultiModalDataset(Dataset):
    """多模态数据集"""
    
    def __init__(self, data_path: str, split: str = 'train', transform=None):
        """
        Args:
            data_path: 数据目录路径
            split: 数据集划分 ('train', 'val', 'test')
            transform: 数据增强
        """
        self.data_path = os.path.join(data_path, split)
        self.transform = transform
        
        # 加载数据
        self.video_features = np.load(os.path.join(self.data_path, 'video_features.npy'))
        self.audio_features = np.load(os.path.join(self.data_path, 'audio_features.npy'))
        self.health_features = np.load(os.path.join(self.data_path, 'health_features.npy'))
        self.medication_features = np.load(os.path.join(self.data_path, 'medication_features.npy'))
        self.labels = np.load(os.path.join(self.data_path, 'labels.npy'))
        
        print(f"加载 {split} 数据集: {len(self.labels)} 个样本")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'video': torch.tensor(self.video_features[idx], dtype=torch.float32),
            'audio': torch.tensor(self.audio_features[idx], dtype=torch.float32),
            'health': torch.tensor(self.health_features[idx], dtype=torch.float32),
            'medication': torch.tensor(self.medication_features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


class MultiModalDatasetFromMemory(Dataset):
    """从内存中的数据创建数据集"""
    
    def __init__(self, data: Dict):
        """
        Args:
            data: 包含所有模态数据的字典
        """
        self.video_features = torch.tensor(data['video_features'], dtype=torch.float32)
        self.audio_features = torch.tensor(data['audio_features'], dtype=torch.float32)
        self.health_features = torch.tensor(data['health_features'], dtype=torch.float32)
        self.medication_features = torch.tensor(data['medication_features'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'video': self.video_features[idx],
            'audio': self.audio_features[idx],
            'health': self.health_features[idx],
            'medication': self.medication_features[idx],
            'label': self.labels[idx],
        }


def create_dataloaders(config, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试数据加载器
    
    Args:
        config: 配置对象
        num_workers: 数据加载线程数
        
    Returns:
        train_loader, val_loader, test_loader
    """
    data_path = config['data']['processed_data_dir']
    batch_size = config['train']['batch_size']
    
    # 创建数据集
    train_dataset = MultiModalDataset(data_path, 'train')
    val_dataset = MultiModalDataset(data_path, 'val')
    test_dataset = MultiModalDataset(data_path, 'test')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_dataloaders_from_splits(splits: Dict, batch_size: int = 32, num_workers: int = 4):
    """
    从数据划分字典创建数据加载器
    
    Args:
        splits: 包含 train, val, test 的数据划分字典
        batch_size: 批次大小
        num_workers: 数据加载线程数
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = MultiModalDatasetFromMemory(splits['train'])
    val_dataset = MultiModalDatasetFromMemory(splits['val'])
    test_dataset = MultiModalDatasetFromMemory(splits['test'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# 测试代码
if __name__ == "__main__":
    # 测试数据加载器
    import sys
    sys.path.append('..')
    from config import get_config
    
    config = get_config()
    
    # 假设数据已经生成
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        
        # 测试一个批次
        for batch in train_loader:
            print(f"视频特征: {batch['video'].shape}")
            print(f"音频特征: {batch['audio'].shape}")
            print(f"生理特征: {batch['health'].shape}")
            print(f"用药特征: {batch['medication'].shape}")
            print(f"标签: {batch['label'].shape}")
            break
    except FileNotFoundError:
        print("数据文件不存在，请先运行 prepare_data.py 生成数据")
