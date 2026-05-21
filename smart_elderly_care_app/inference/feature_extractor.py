"""
特征提取模块
将用户输入（视频、音频、生理数据、用药信息）转换为模型可接受的特征向量
"""

import os
import numpy as np
from typing import Optional, Dict, List
import cv2
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FEATURE_DIMS


class VideoFeatureExtractor:
    """视频特征提取器"""
    
    def __init__(self, target_dim: int = 768):
        self.target_dim = target_dim
        # 注意：实际部署时应加载预训练的VideoMAE模型
        # 这里使用简化版本
    
    def extract_from_file(self, video_path: str) -> np.ndarray:
        """
        从视频文件提取特征
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            768维特征向量
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 读取视频
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"无法读取视频帧: {video_path}")
        
        return self.extract_from_frames(frames)
    
    def extract_from_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        从帧列表提取特征
        
        Args:
            frames: 视频帧列表
            
        Returns:
            768维特征向量
        """
        # 简化版：使用帧的统计特征
        # 实际部署时应使用VideoMAE等预训练模型
        
        # 均匀采样16帧
        sample_indices = np.linspace(0, len(frames) - 1, min(16, len(frames)), dtype=int)
        sampled_frames = [frames[i] for i in sample_indices]
        
        # 提取每帧的特征
        frame_features = []
        for frame in sampled_frames:
            # 转换为灰度图并调整大小
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (224, 224))
            
            # 计算统计特征
            mean = np.mean(resized)
            std = np.std(resized)
            hist = np.histogram(resized, bins=32, range=(0, 256))[0]
            hist = hist / (hist.sum() + 1e-6)
            
            frame_features.append(np.concatenate([[mean, std], hist]))
        
        # 聚合为视频级特征
        video_feature = np.mean(frame_features, axis=0)
        
        # 投影到目标维度
        feature = self._project_to_dim(video_feature, self.target_dim)
        
        return feature.astype(np.float32)
    
    def _project_to_dim(self, feature: np.ndarray, target_dim: int) -> np.ndarray:
        """投影到目标维度"""
        current_dim = len(feature)
        
        if current_dim < target_dim:
            # 重复填充
            repeat_times = target_dim // current_dim + 1
            projected = np.tile(feature, repeat_times)[:target_dim]
        elif current_dim > target_dim:
            # 截断
            projected = feature[:target_dim]
        else:
            projected = feature
        
        # 添加随机噪声模拟深度特征
        projected = projected + np.random.randn(target_dim) * 0.1
        
        return projected


class AudioFeatureExtractor:
    """音频特征提取器"""
    
    def __init__(self, target_dim: int = 768, sample_rate: int = 16000):
        self.target_dim = target_dim
        self.sample_rate = sample_rate
    
    def extract_from_file(self, audio_path: str) -> np.ndarray:
        """
        从音频文件提取特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            768维特征向量
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        try:
            import librosa
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            return self.extract_from_signal(y)
        except ImportError:
            # librosa未安装，使用简化方法
            return self._extract_simple(audio_path)
    
    def extract_from_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        从音频信号提取特征
        
        Args:
            signal: 音频信号数组
            
        Returns:
            768维特征向量
        """
        try:
            import librosa
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(y=signal, sr=self.sample_rate, n_mfcc=40)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # 提取Mel频谱
            mel = librosa.feature.melspectrogram(y=signal, sr=self.sample_rate, n_mels=64)
            mel_mean = np.mean(mel, axis=1)
            
            # 提取过零率
            zcr = librosa.feature.zero_crossing_rate(signal)
            zcr_mean = np.mean(zcr)
            
            # 提取能量
            rms = librosa.feature.rms(y=signal)
            rms_mean = np.mean(rms)
            
            # 合并特征
            feature = np.concatenate([
                mfcc_mean, mfcc_std, mel_mean, [zcr_mean, rms_mean]
            ])
            
            # 投影到目标维度
            feature = self._project_to_dim(feature, self.target_dim)
            
        except ImportError:
            feature = np.random.randn(self.target_dim).astype(np.float32) * 0.1
        
        return feature.astype(np.float32)
    
    def _extract_simple(self, audio_path: str) -> np.ndarray:
        """简化版特征提取"""
        # 使用随机特征模拟
        return np.random.randn(self.target_dim).astype(np.float32) * 0.1
    
    def _project_to_dim(self, feature: np.ndarray, target_dim: int) -> np.ndarray:
        """投影到目标维度"""
        current_dim = len(feature)
        
        if current_dim < target_dim:
            repeat_times = target_dim // current_dim + 1
            projected = np.tile(feature, repeat_times)[:target_dim]
        elif current_dim > target_dim:
            projected = feature[:target_dim]
        else:
            projected = feature
        
        projected = projected + np.random.randn(target_dim) * 0.05
        
        return projected


class HealthFeatureExtractor:
    """生理数据特征提取器"""
    
    def __init__(self, target_dim: int = 256):
        self.target_dim = target_dim
        self.feature_names = ["heart_rate", "blood_oxygen", "systolic", "diastolic", "steps"]
    
    def extract_from_dict(self, health_data: Dict) -> np.ndarray:
        """
        从字典数据提取特征
        
        Args:
            health_data: 包含生理数据的字典
                {
                    "heart_rate": 75,
                    "blood_oxygen": 97,
                    "systolic": 120,
                    "diastolic": 80,
                    "steps": 3000
                }
            
        Returns:
            256维特征向量
        """
        features = []
        
        for name in self.feature_names:
            value = health_data.get(name)
            if value is not None:
                features.append(float(value))
            else:
                # 使用默认值
                default_values = {
                    "heart_rate": 75,
                    "blood_oxygen": 97,
                    "systolic": 120,
                    "diastolic": 80,
                    "steps": 2000
                }
                features.append(default_values.get(name, 0))
        
        # 扩展到目标维度
        feature = self._expand_features(np.array(features))
        
        return feature.astype(np.float32)
    
    def extract_from_series(self, time_series: np.ndarray) -> np.ndarray:
        """
        从时间序列提取特征
        
        Args:
            time_series: [seq_len, num_features] 的时间序列
            
        Returns:
            256维特征向量
        """
        if len(time_series.shape) == 1:
            time_series = time_series.reshape(-1, 1)
        
        # 计算统计特征
        mean = np.mean(time_series, axis=0)
        std = np.std(time_series, axis=0)
        max_val = np.max(time_series, axis=0)
        min_val = np.min(time_series, axis=0)
        
        # 合并
        features = np.concatenate([mean, std, max_val, min_val])
        
        # 扩展到目标维度
        feature = self._expand_features(features)
        
        return feature.astype(np.float32)
    
    def _expand_features(self, features: np.ndarray) -> np.ndarray:
        """扩展特征到目标维度"""
        current_dim = len(features)
        
        if current_dim < self.target_dim:
            # 使用重复和噪声扩展
            repeat_times = self.target_dim // current_dim + 1
            expanded = np.tile(features, repeat_times)[:self.target_dim]
            # 添加小噪声
            expanded = expanded + np.random.randn(self.target_dim) * 0.01
        else:
            expanded = features[:self.target_dim]
        
        return expanded


class MedicationFeatureExtractor:
    """用药数据特征提取器"""
    
    def __init__(self, target_dim: int = 128):
        self.target_dim = target_dim
    
    def extract_from_records(self, medication_records: List[Dict]) -> np.ndarray:
        """
        从用药记录提取特征
        
        Args:
            medication_records: 用药记录列表
                [
                    {"name": "降压药", "dosage": "1片", "time": "08:00", "taken": True},
                    ...
                ]
            
        Returns:
            128维特征向量
        """
        features = []
        
        # 用药数量
        features.append(len(medication_records))
        
        # 服药率
        if len(medication_records) > 0:
            taken_count = sum(1 for r in medication_records if r.get("taken", False))
            features.append(taken_count / len(medication_records))
        else:
            features.append(0)
        
        # 漏服数量
        missed = sum(1 for r in medication_records if not r.get("taken", False))
        features.append(missed)
        
        # 扩展到目标维度
        feature = self._expand_features(np.array(features))
        
        return feature.astype(np.float32)
    
    def extract_from_dict(self, medication_data: Dict) -> np.ndarray:
        """
        从字典数据提取特征
        
        Args:
            medication_data: 用药数据字典
                {
                    "total_medications": 3,
                    "adherence_rate": 0.85,
                    "missed_doses": 2
                }
            
        Returns:
            128维特征向量
        """
        features = [
            medication_data.get("total_medications", 0),
            medication_data.get("adherence_rate", 0),
            medication_data.get("missed_doses", 0),
        ]
        
        feature = self._expand_features(np.array(features))
        
        return feature.astype(np.float32)
    
    def _expand_features(self, features: np.ndarray) -> np.ndarray:
        """扩展特征到目标维度"""
        current_dim = len(features)
        
        if current_dim < self.target_dim:
            repeat_times = self.target_dim // current_dim + 1
            expanded = np.tile(features, repeat_times)[:self.target_dim]
            expanded = expanded + np.random.randn(self.target_dim) * 0.01
        else:
            expanded = features[:self.target_dim]
        
        return expanded


class MultiModalFeatureExtractor:
    """多模态特征提取器（统一接口）"""
    
    def __init__(self):
        self.video_extractor = VideoFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        self.health_extractor = HealthFeatureExtractor()
        self.medication_extractor = MedicationFeatureExtractor()
    
    def extract_all(self, data: Dict) -> Dict[str, np.ndarray]:
        """
        提取所有模态特征
        
        Args:
            data: 包含各模态数据的字典
                {
                    "video_path": "path/to/video.mp4",  # 或 "video_frames": [...]
                    "audio_path": "path/to/audio.wav",  # 或 "audio_signal": [...]
                    "health_data": {"heart_rate": 75, ...},
                    "medication_records": [...]
                }
        
        Returns:
            各模态特征字典
        """
        features = {}
        
        # 视频特征
        if "video_path" in data and data["video_path"]:
            try:
                features["video"] = self.video_extractor.extract_from_file(data["video_path"])
            except Exception as e:
                print(f"视频特征提取失败: {e}")
                features["video"] = None
        elif "video_frames" in data:
            features["video"] = self.video_extractor.extract_from_frames(data["video_frames"])
        else:
            features["video"] = None
        
        # 音频特征
        if "audio_path" in data and data["audio_path"]:
            try:
                features["audio"] = self.audio_extractor.extract_from_file(data["audio_path"])
            except Exception as e:
                print(f"音频特征提取失败: {e}")
                features["audio"] = None
        elif "audio_signal" in data:
            features["audio"] = self.audio_extractor.extract_from_signal(data["audio_signal"])
        else:
            features["audio"] = None
        
        # 生理特征
        if "health_data" in data:
            features["health"] = self.health_extractor.extract_from_dict(data["health_data"])
        else:
            features["health"] = None
        
        # 用药特征
        if "medication_records" in data:
            features["medication"] = self.medication_extractor.extract_from_records(data["medication_records"])
        elif "medication_data" in data:
            features["medication"] = self.medication_extractor.extract_from_dict(data["medication_data"])
        else:
            features["medication"] = None
        
        return features


if __name__ == "__main__":
    # 测试特征提取
    extractor = MultiModalFeatureExtractor()
    
    # 测试生理数据提取
    health_data = {
        "heart_rate": 75,
        "blood_oxygen": 97,
        "systolic": 120,
        "diastolic": 80,
        "steps": 3000
    }
    health_feature = extractor.health_extractor.extract_from_dict(health_data)
    print(f"生理特征: {health_feature.shape}")
    
    # 测试用药数据提取
    med_data = {
        "total_medications": 3,
        "adherence_rate": 0.85,
        "missed_doses": 2
    }
    med_feature = extractor.medication_extractor.extract_from_dict(med_data)
    print(f"用药特征: {med_feature.shape}")
