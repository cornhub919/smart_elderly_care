"""
音频异常检测模块
检测呼救声、撞击声等异常音频事件
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AudioEvent:
    """音频事件数据类"""
    timestamp: float
    event_type: str
    confidence: float
    duration: float
    description: str


class AudioDetector:
    """音频异常检测器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {
            "sample_rate": 16000,
            "frame_length": 1024,
            "hop_length": 512,
            "n_mels": 128,
            "help_keywords": ["救命", "帮帮我", "我摔倒了", "难受", "过来一下", "疼"],
            "impact_labels": ["impact", "crash", "bang", "fall", "glass"],
        }
        
        self.sample_rate = self.config["sample_rate"]
        
        # 音频事件类型定义
        self.event_types = {
            "help_call": {
                "name": "呼救声",
                "keywords": self.config["help_keywords"],
                "threshold": 0.6,
            },
            "impact": {
                "name": "撞击声",
                "frequency_range": (100, 2000),
                "threshold": 0.5,
            },
            "fall_sound": {
                "name": "摔倒声",
                "frequency_range": (50, 500),
                "threshold": 0.5,
            },
            "scream": {
                "name": "尖叫声",
                "frequency_range": (1000, 4000),
                "threshold": 0.6,
            },
            "cough": {
                "name": "咳嗽声",
                "frequency_range": (200, 800),
                "threshold": 0.4,
            },
        }
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """加载音频文件"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            print(f"加载音频失败: {e}")
            return np.array([]), self.sample_rate
    
    def extract_features(self, audio: np.ndarray) -> dict:
        """提取音频特征"""
        features = {}
        
        # 1. Mel频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_fft=self.config["frame_length"],
            hop_length=self.config["hop_length"],
            n_mels=self.config["n_mels"]
        )
        features["mel_spectrogram"] = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 2. MFCC特征
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate,
            n_mfcc=13,
            n_fft=self.config["frame_length"],
            hop_length=self.config["hop_length"]
        )
        features["mfcc"] = mfcc
        
        # 3. 能量特征
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.config["frame_length"],
            hop_length=self.config["hop_length"]
        )
        features["rms"] = rms
        
        # 4. 过零率
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.config["frame_length"],
            hop_length=self.config["hop_length"]
        )
        features["zcr"] = zcr
        
        # 5. 频谱质心
        cent = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.config["frame_length"],
            hop_length=self.config["hop_length"]
        )
        features["spectral_centroid"] = cent
        
        # 6. 频谱带宽
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.config["frame_length"],
            hop_length=self.config["hop_length"]
        )
        features["spectral_bandwidth"] = bandwidth
        
        return features
    
    def detect_impact_sound(self, audio: np.ndarray, features: dict) -> Tuple[bool, float]:
        """检测撞击声"""
        # 撞击声特征：短时高能量、宽频带
        rms = features["rms"][0]
        bandwidth = features["spectral_bandwidth"][0]
        
        # 计算能量峰值
        energy_threshold = np.mean(rms) + 2 * np.std(rms)
        peak_indices = np.where(rms > energy_threshold)[0]
        
        if len(peak_indices) > 0:
            # 检查是否有突然的能量上升
            for idx in peak_indices:
                if idx > 0:
                    energy_ratio = rms[idx] / (rms[idx-1] + 1e-6)
                    if energy_ratio > 3:  # 能量突然增加3倍以上
                        return True, min(energy_ratio / 10, 1.0)
        
        return False, 0.0
    
    def detect_scream(self, audio: np.ndarray, features: dict) -> Tuple[bool, float]:
        """检测尖叫声"""
        # 尖叫声特征：高频、高能量、高过零率
        cent = features["spectral_centroid"][0]
        zcr = features["zcr"][0]
        rms = features["rms"][0]
        
        # 尖叫声通常有较高的频谱质心
        high_freq_ratio = np.mean(cent > 2000)
        high_zcr_ratio = np.mean(zcr > 0.1)
        high_energy_ratio = np.mean(rms > np.mean(rms) + np.std(rms))
        
        confidence = (high_freq_ratio * 0.4 + high_zcr_ratio * 0.3 + high_energy_ratio * 0.3)
        
        if confidence > 0.5:
            return True, confidence
        
        return False, 0.0
    
    def detect_cough(self, audio: np.ndarray, features: dict) -> Tuple[bool, float]:
        """检测咳嗽声"""
        # 咳嗽声特征：短促爆发、中频能量集中
        rms = features["rms"][0]
        cent = features["spectral_centroid"][0]
        
        # 咳嗽声通常有短促的能量爆发
        energy_peaks = []
        for i in range(1, len(rms) - 1):
            if rms[i] > rms[i-1] and rms[i] > rms[i+1]:
                if rms[i] > np.mean(rms) + np.std(rms):
                    energy_peaks.append(i)
        
        # 检查频谱质心是否在中频范围
        if len(energy_peaks) > 0:
            mid_freq_ratio = np.mean((cent > 200) & (cent < 800))
            if mid_freq_ratio > 0.3:
                return True, min(len(energy_peaks) / 10, 1.0)
        
        return False, 0.0
    
    def detect_silence_anomaly(self, audio: np.ndarray, 
                               threshold_duration: float = 10.0) -> Tuple[bool, float]:
        """检测异常静默（长时间无声音）"""
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.config["frame_length"],
            hop_length=self.config["hop_length"]
        )[0]
        
        # 静默阈值
        silence_threshold = 0.01
        silent_frames = np.sum(rms < silence_threshold)
        silent_duration = silent_frames * self.config["hop_length"] / self.sample_rate
        
        if silent_duration > threshold_duration:
            return True, min(silent_duration / threshold_duration, 1.0)
        
        return False, 0.0
    
    def analyze_audio(self, audio: np.ndarray) -> Dict:
        """分析音频，返回所有检测结果"""
        features = self.extract_features(audio)
        
        results = {
            "duration": len(audio) / self.sample_rate,
            "events": [],
            "features_summary": {
                "mean_energy": float(np.mean(features["rms"])),
                "mean_centroid": float(np.mean(features["spectral_centroid"])),
                "mean_zcr": float(np.mean(features["zcr"])),
            }
        }
        
        # 检测各类事件
        # 1. 撞击声
        is_impact, impact_conf = self.detect_impact_sound(audio, features)
        if is_impact:
            results["events"].append(AudioEvent(
                timestamp=0,
                event_type="impact",
                confidence=impact_conf,
                duration=results["duration"],
                description="检测到撞击声"
            ))
        
        # 2. 尖叫声
        is_scream, scream_conf = self.detect_scream(audio, features)
        if is_scream:
            results["events"].append(AudioEvent(
                timestamp=0,
                event_type="scream",
                confidence=scream_conf,
                duration=results["duration"],
                description="检测到尖叫声"
            ))
        
        # 3. 咳嗽声
        is_cough, cough_conf = self.detect_cough(audio, features)
        if is_cough:
            results["events"].append(AudioEvent(
                timestamp=0,
                event_type="cough",
                confidence=cough_conf,
                duration=results["duration"],
                description="检测到咳嗽声"
            ))
        
        # 4. 异常静默
        is_silent, silent_conf = self.detect_silence_anomaly(audio)
        if is_silent:
            results["events"].append(AudioEvent(
                timestamp=0,
                event_type="silence",
                confidence=silent_conf,
                duration=results["duration"],
                description="检测到异常静默"
            ))
        
        return results
    
    def detect_from_file(self, audio_path: str) -> Dict:
        """从文件检测音频事件"""
        audio, sr = self.load_audio(audio_path)
        if len(audio) == 0:
            return {"error": "无法加载音频文件", "events": []}
        
        return self.analyze_audio(audio)


class KeywordDetector:
    """关键词检测器（简化版，用于呼救识别）"""
    
    def __init__(self, keywords: List[str] = None):
        self.keywords = keywords or ["救命", "帮帮我", "我摔倒了", "难受", "过来一下", "疼"]
        
        # 注意：实际应用中需要使用ASR模型（如Whisper）
        # 这里提供一个接口框架
        self.asr_model = None
    
    def load_asr_model(self):
        """加载ASR模型（需要安装whisper）"""
        try:
            import whisper
            self.asr_model = whisper.load_model("base")
            return True
        except ImportError:
            print("警告：未安装whisper，关键词检测功能受限")
            return False
    
    def detect_keywords(self, audio_path: str) -> List[Dict]:
        """检测关键词"""
        if self.asr_model is None:
            return []
        
        try:
            result = self.asr_model.transcribe(audio_path, language="zh")
            text = result["text"]
            
            detected = []
            for keyword in self.keywords:
                if keyword in text:
                    detected.append({
                        "keyword": keyword,
                        "text": text,
                        "confidence": 0.8  # 简化置信度
                    })
            
            return detected
        except Exception as e:
            print(f"关键词检测失败: {e}")
            return []


if __name__ == "__main__":
    # 测试代码
    print("音频检测模块测试")
    detector = AudioDetector()
    print("模块初始化成功")
    
    # 测试特征提取
    test_audio = np.random.randn(16000)  # 1秒随机音频
    features = detector.extract_features(test_audio)
    print(f"提取特征: {list(features.keys())}")
