"""
特征提取模块（统一真源版）
========================
迭代1 改造：本文件不再各自实现特征提取，而是代理到 V2 的统一特征提取器
(models.unified_feature_extractor)，保证训练特征与推理特征分布一致。

对外保留原类名与方法签名（app.py / predictor.py 依赖），内部委托实现：
  - VideoFeatureExtractor       → RealVideoFeatureExtractor
  - AudioFeatureExtractor       → RealAudioFeatureExtractor
  - HealthFeatureExtractor      → RealHealthFeatureExtractor
  - MedicationFeatureExtractor  → RealMedicationFeatureExtractor
  - MultiModalFeatureExtractor  → UnifiedFeatureExtractor
"""

import os
import sys
import numpy as np
from typing import Optional, Dict, List

# ---- 注入 V2 路径，导入统一提取器（单一真源）----
_THIS = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_THIS)
_PROJECT_ROOT = os.path.dirname(_APP_ROOT)
_V2_ROOT = os.path.join(_PROJECT_ROOT, "smart_elderly_care_v2")
for _p in (_PROJECT_ROOT, _V2_ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from models.unified_feature_extractor import (  # noqa: E402
    UnifiedFeatureExtractor as _UnifiedExtractor,
    RealVideoFeatureExtractor as _RealVideo,
    RealAudioFeatureExtractor as _RealAudio,
    RealHealthFeatureExtractor as _RealHealth,
    RealMedicationFeatureExtractor as _RealMed,
    VIDEO_DIM, AUDIO_DIM, HEALTH_DIM, MED_DIM,
)


class VideoFeatureExtractor:
    """视频特征提取器（代理到统一真源）"""

    def __init__(self, target_dim: int = VIDEO_DIM):
        self.target_dim = target_dim
        self._impl = _RealVideo(target_dim=target_dim)

    def extract_from_file(self, video_path: str) -> np.ndarray:
        return self._impl.extract_from_file(video_path)

    def extract_from_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        return self._impl.extract_from_frames(frames)

    def _project_to_dim(self, feature: np.ndarray, target_dim: int) -> np.ndarray:
        # 保留旧方法名以兼容，但实际由统一提取器内部确定性投影处理
        return feature


class AudioFeatureExtractor:
    """音频特征提取器（代理到统一真源）"""

    def __init__(self, target_dim: int = AUDIO_DIM, sample_rate: int = 16000):
        self.target_dim = target_dim
        self.sample_rate = sample_rate
        self._impl = _RealAudio(target_dim=target_dim, sample_rate=sample_rate)

    def extract_from_file(self, audio_path: str) -> np.ndarray:
        return self._impl.extract_from_file(audio_path)

    def extract_from_signal(self, signal: np.ndarray) -> np.ndarray:
        return self._impl.extract_from_signal(signal)

    def _extract_simple(self, audio_path: str) -> np.ndarray:
        # 兼容旧接口：librosa 不可用时统一提取器内部已处理
        return self._impl.extract_from_file(audio_path)

    def _project_to_dim(self, feature: np.ndarray, target_dim: int) -> np.ndarray:
        return feature


class HealthFeatureExtractor:
    """生理数据特征提取器（代理到统一真源）"""

    def __init__(self, target_dim: int = HEALTH_DIM):
        self.target_dim = target_dim
        self.feature_names = ["heart_rate", "blood_oxygen", "systolic", "diastolic", "steps"]
        self._impl = _RealHealth(target_dim=target_dim)

    def extract_from_dict(self, health_data: Dict) -> np.ndarray:
        return self._impl.extract_from_dict(health_data)

    def extract_from_series(self, time_series: np.ndarray) -> np.ndarray:
        return self._impl.extract_from_series(time_series)

    def _expand_features(self, features: np.ndarray) -> np.ndarray:
        return features


class MedicationFeatureExtractor:
    """用药数据特征提取器（代理到统一真源）"""

    def __init__(self, target_dim: int = MED_DIM):
        self.target_dim = target_dim
        self._impl = _RealMed(target_dim=target_dim)

    def extract_from_records(self, medication_records: List[Dict]) -> np.ndarray:
        return self._impl.extract_from_records(medication_records)

    def extract_from_dict(self, medication_data: Dict) -> np.ndarray:
        return self._impl.extract_from_dict(medication_data)

    def _expand_features(self, features: np.ndarray) -> np.ndarray:
        return features


class MultiModalFeatureExtractor:
    """多模态特征提取器（统一接口，代理到统一真源）"""

    def __init__(self):
        self._impl = _UnifiedExtractor()
        # 保留子提取器引用以兼容旧调用
        self.video_extractor = VideoFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        self.health_extractor = HealthFeatureExtractor()
        self.medication_extractor = MedicationFeatureExtractor()

    def extract_all(self, data: Dict) -> Dict[str, np.ndarray]:
        """
        提取所有模态特征。

        Args:
            data: {
                "video_path" / "video_frames": ...,
                "audio_path" / "audio_signal": ...,
                "health_data" / "health_series": ...,
                "medication_records" / "medication_data": ...,
            }
        Returns:
            {"video": (768,), "audio": (768,), "health": (256,), "medication": (128,)}
            缺失模态返回 None
        """
        return self._impl.extract_all(data)


if __name__ == "__main__":
    # 自测
    print("=== App 特征提取器（统一真源版）自测 ===")
    ext = MultiModalFeatureExtractor()

    health_data = {"heart_rate": 75, "blood_oxygen": 97,
                   "systolic": 120, "diastolic": 80, "steps": 3000}
    hf = ext.health_extractor.extract_from_dict(health_data)
    print(f"生理特征: {hf.shape}")

    med_data = {"total_medications": 3, "adherence_rate": 0.85, "missed_doses": 2}
    mf = ext.medication_extractor.extract_from_dict(med_data)
    print(f"用药特征: {mf.shape}")
    print("[PASS]")
