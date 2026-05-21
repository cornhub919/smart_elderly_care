"""
缺失数据处理模块
当用户未提供某些模态数据时，使用预定义的默认特征
"""

import os
import numpy as np
from typing import Optional, Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_FEATURES, FEATURE_DIMS


class MissingDataHandler:
    """缺失数据处理器"""
    
    def __init__(self):
        self.default_features = {}
        self._load_defaults()
    
    def _load_defaults(self):
        """加载默认特征"""
        for modality, path in DEFAULT_FEATURES.items():
            if os.path.exists(path):
                self.default_features[modality] = np.load(path)
            else:
                # 如果默认文件不存在，生成并保存
                self.default_features[modality] = self._generate_default(modality)
                self._save_default(modality)
    
    def _generate_default(self, modality: str) -> np.ndarray:
        """生成默认特征"""
        dim = FEATURE_DIMS[modality]
        
        if modality == "video":
            # 正常活动状态的特征（模拟）
            # 特征分布较为均匀，表示正常的活动状态
            return np.random.randn(dim).astype(np.float32) * 0.1
        
        elif modality == "audio":
            # 安静环境的特征（模拟）
            # 特征值较小，表示安静环境
            return np.random.randn(dim).astype(np.float32) * 0.05
        
        elif modality == "health":
            # 健康基线特征（模拟）
            # 表示正常健康状态
            return np.random.randn(dim).astype(np.float32) * 0.1
        
        elif modality == "medication":
            # 无用药信息
            # 零向量表示无用药记录
            return np.zeros(dim, dtype=np.float32)
        
        return np.zeros(dim, dtype=np.float32)
    
    def _save_default(self, modality: str):
        """保存默认特征"""
        os.makedirs(os.path.dirname(DEFAULT_FEATURES[modality]), exist_ok=True)
        np.save(DEFAULT_FEATURES[modality], self.default_features[modality])
    
    def handle_missing(self, features: Optional[np.ndarray], modality: str) -> np.ndarray:
        """
        处理缺失数据
        
        Args:
            features: 用户提供的特征，可能为None
            modality: 模态名称
            
        Returns:
            处理后的特征向量
        """
        if features is not None and len(features) > 0:
            # 验证维度
            expected_dim = FEATURE_DIMS[modality]
            if len(features) != expected_dim:
                # 维度不匹配，使用默认
                print(f"警告: {modality}特征维度不匹配，期望{expected_dim}，实际{len(features)}，使用默认特征")
                return self.default_features[modality]
            return features
        
        # 使用默认特征
        return self.default_features[modality]
    
    def handle_batch(self, features_dict: Dict[str, Optional[np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        批量处理缺失数据
        
        Args:
            features_dict: 各模态特征字典
            
        Returns:
            处理后的特征字典
        """
        result = {}
        for modality in FEATURE_DIMS.keys():
            result[modality] = self.handle_missing(
                features_dict.get(modality), 
                modality
            )
        return result
    
    def get_missing_info(self, features_dict: Dict[str, Optional[np.ndarray]]) -> Dict[str, bool]:
        """
        获取缺失信息
        
        Args:
            features_dict: 各模态特征字典
            
        Returns:
            各模态是否缺失的字典
        """
        missing_info = {}
        for modality in FEATURE_DIMS.keys():
            features = features_dict.get(modality)
            missing_info[modality] = (features is None or len(features) == 0)
        return missing_info


def generate_default_files():
    """生成所有默认特征文件"""
    handler = MissingDataHandler()
    print("默认特征文件已生成:")
    for modality, path in DEFAULT_FEATURES.items():
        print(f"  {modality}: {path}")


if __name__ == "__main__":
    generate_default_files()
