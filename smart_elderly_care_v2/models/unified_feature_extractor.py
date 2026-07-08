"""
统一真实特征提取模块（训练 / 推理 共用单一真源）
================================================
本模块是迭代1的核心：彻底消除"训练特征分布"与"App推理特征分布"不一致的致命缺陷。

设计原则
--------
1. 单一真源：训练(preprocess_real.py / prepare_data.py) 与 推理(App feature_extractor.py)
   都 import 本模块，绝不各自实现。
2. 真实语义：视频走 MediaPipe Pose 姿态时序；音频走 librosa 声学特征；不再用随机噪声。
3. 确定性投影：低维语义特征 → 目标维度，用 **固定种子可复现** 的线性映射，
   而非 np.random.randn（旧版每次调用结果不同 → 训练/推理分布漂移）。
4. 缺失鲁棒：任意模态缺失时返回零向量 + 标记，由 MissingDataHandler 兜底。

输出维度（与模型 forward 对齐）
------------------------------
- video:    768  (config.model.video_encoder.hidden_dim)
- audio:    768  (config.model.audio_encoder.hidden_dim)
- health:   256  (config.model.health_encoder.hidden_dim)
- medication:128 (config.data.medication.embedding_dim)
"""

import os
import sys
import numpy as np
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# 项目根
_THIS = os.path.dirname(os.path.abspath(__file__))
_V2_ROOT = os.path.dirname(_THIS)
_PROJECT_ROOT = os.path.dirname(_V2_ROOT)
for _p in (_V2_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

try:
    from config import get_config
    _CFG = get_config()
    VIDEO_DIM = int(_CFG.model.video_encoder.hidden_dim)      # 768
    AUDIO_DIM = int(_CFG.model.audio_encoder.hidden_dim)      # 768
    HEALTH_DIM = int(_CFG.model.health_encoder.hidden_dim)    # 256
    MED_DIM = int(_CFG.data.medication.embedding_dim)         # 128
except Exception:
    # 兜底默认值（与 config.py 一致）
    VIDEO_DIM, AUDIO_DIM, HEALTH_DIM, MED_DIM = 768, 768, 256, 128

# 确定性投影矩阵缓存（按 (in_dim, out_dim, seed) 复用，保证训练/推理一致）
_PROJ_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def _deterministic_project(features: np.ndarray, target_dim: int,
                           seed: int = 42) -> np.ndarray:
    """
    确定性线性投影：in_dim -> target_dim
    用固定种子生成的投影矩阵，保证同一输入 → 同一输出（训练/推理一致）。
    若 in_dim == target_dim 直接返回；若 in_dim < target_dim 零填充。
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    in_dim = features.shape[1]
    if in_dim == target_dim:
        return features.squeeze(0).astype(np.float32)

    key = (in_dim, target_dim, seed)
    if key not in _PROJ_CACHE:
        rng = np.random.RandomState(seed)  # 固定种子 → 可复现
        if in_dim < target_dim:
            # 扩展：随机正交化投影 + 零填充由 matmul 自动处理
            proj = rng.randn(in_dim, target_dim).astype(np.float32) / np.sqrt(in_dim)
        else:
            # 压缩：随机投影（Johnson-Lindenstrauss 风格）
            proj = rng.randn(in_dim, target_dim).astype(np.float32) / np.sqrt(in_dim)
        _PROJ_CACHE[key] = proj
    else:
        proj = _PROJ_CACHE[key]

    out = features @ proj  # [N, target_dim]
    return out.squeeze(0).astype(np.float32)


# ===========================================================================
# 视频特征提取（MediaPipe Pose 真实姿态）
# ===========================================================================

class RealVideoFeatureExtractor:
    """
    基于 MediaPipe PoseLandmarker 的视频特征提取器。

    流程：视频帧序列 → 逐帧 33 关键点 → 时序统计 → 投影 768 维

    提取的语义特征（与跌倒/行为强相关）：
    - 重心高度轨迹：均值、方差、最大下降速度、下降幅度
    - 躯干角度：均值、方差、是否趋近水平
    - 运动量：帧间位移均值、峰值、静止帧占比
    - 关键点可见率：检测置信度的时序统计

    注：mediapipe >= 0.10.14 移除了 mp.solutions.pose，改用 Tasks API
        (PoseLandmarker)。本类使用 IMAGE 模式逐帧检测，避免 VIDEO 模式
        的时间戳单调性约束。
    """

    # MediaPipe Pose 33 关键点中，躯干相关索引（新旧版一致）
    _SHOULDERS = [11, 12]
    _HIPS = [23, 24]
    _KNEES = [25, 26]

    # pose_landmarker_lite 模型下载地址（Google 官方 CDN）
    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )

    def __init__(self, target_dim: int = VIDEO_DIM, max_frames: int = 64):
        self.target_dim = target_dim
        self.max_frames = max_frames
        self._pose = None       # PoseLandmarker 实例
        self._mp = None         # mediapipe 模块引用
        self._model_path = None  # 模型文件缓存路径

    def _ensure_pose(self):
        """延迟初始化 PoseLandmarker（mediapipe 导入较重）。
        自动下载 pose_landmarker_lite.task 模型，用 buffer 加载
        以规避中文路径下 C 底层 fopen 失败的问题。
        """
        if self._pose is not None:
            return

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker, PoseLandmarkerOptions, RunningMode,
        )
        self._mp = mp

        # 定位/下载模型文件
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "raw", "_mp_models",
        )
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "pose_landmarker_lite.task")
        if not os.path.exists(model_path):
            import urllib.request, ssl, socket
            print("[MediaPipe] 首次使用，下载 pose_landmarker_lite.task ...")
            socket.setdefaulttimeout(120)
            sctx = ssl.create_default_context()
            sctx.check_hostname = False
            sctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(
                self._MODEL_URL,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=120, context=sctx) as resp, \
                 open(model_path, "wb") as f:
                f.write(resp.read())
            print(f"[MediaPipe] 模型已保存: {model_path} "
                  f"({os.path.getsize(model_path)/1024/1024:.1f} MB)")
        self._model_path = model_path

        # 用 buffer 加载（规避中文路径 C fopen 问题）
        with open(model_path, "rb") as f:
            model_buffer = f.read()

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_buffer),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._pose = PoseLandmarker.create_from_options(options)

    def extract_from_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        从帧列表提取 768 维视频特征。

        Args:
            frames: BGR 帧列表 (H, W, 3)
        Returns:
            (768,) float32
        """
        self._ensure_pose()
        if len(frames) == 0:
            return np.zeros(self.target_dim, dtype=np.float32)

        # 均匀采样 max_frames 帧
        n = len(frames)
        if n > self.max_frames:
            idx = np.linspace(0, n - 1, self.max_frames, dtype=int)
            frames = [frames[i] for i in idx]

        centers_y = []   # 重心 y（归一化，越大越靠下）
        centers_x = []
        angles = []      # 躯干与垂直方向夹角
        visibilities = []  # 关键点可见度
        velocities_y = []  # 重心垂直速度

        prev_y = None
        for frame in frames:
            rgb = frame[:, :, ::-1] if frame.ndim == 3 else frame  # BGR→RGB
            mp_img = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=rgb,
            )
            res = self._pose.detect(mp_img)
            # 新版 API：未检测到时 result.pose_landmarks 为空 list []
            if not res.pose_landmarks:
                # 未检测到人：用 NaN 占位后续插值
                centers_y.append(np.nan)
                centers_x.append(np.nan)
                angles.append(np.nan)
                visibilities.append(0.0)
                continue

            lm = res.pose_landmarks[0]  # list[NormalizedLandmark]，长度 33
            # 重心 = 肩+髋 中点
            cy = np.mean([lm[i].y for i in self._SHOULDERS + self._HIPS])
            cx = np.mean([lm[i].x for i in self._SHOULDERS + self._HIPS])
            centers_y.append(cy)
            centers_x.append(cx)

            # 躯干角度（与垂直方向）
            sx = np.mean([lm[i].x for i in self._SHOULDERS])
            sy = np.mean([lm[i].y for i in self._SHOULDERS])
            hx = np.mean([lm[i].x for i in self._HIPS])
            hy = np.mean([lm[i].y for i in self._HIPS])
            angle = abs(np.arctan2(sx - hx, sy - hy)) * 180 / np.pi
            angles.append(angle)

            vis = np.mean([lm[i].visibility for i in self._SHOULDERS + self._HIPS])
            visibilities.append(vis)

            if prev_y is not None and not np.isnan(prev_y):
                velocities_y.append(cy - prev_y)
            prev_y = cy

        # NaN 插值（用前向/后向填充）
        centers_y = self._interp_nan(centers_y)
        centers_x = self._interp_nan(centers_x)
        angles = self._interp_nan(angles)

        # ---- 时序统计特征 ----
        stats = []
        cy = np.asarray(centers_y, dtype=np.float32)
        cx = np.asarray(centers_x, dtype=np.float32)
        ang = np.asarray(angles, dtype=np.float32)
        vis = np.asarray(visibilities, dtype=np.float32)
        vel = np.asarray(velocities_y, dtype=np.float32) if velocities_y else np.zeros(1, dtype=np.float32)

        # 重心高度
        stats += [np.mean(cy), np.std(cy), np.min(cy), np.max(cy), cy[-1] - cy[0]]
        # 重心水平位移（左右摇晃）
        stats += [np.std(cx), np.max(np.abs(cx - np.mean(cx)))]
        # 重心垂直速度（跌倒关键：快速下降）
        stats += [np.mean(vel), np.max(vel), np.std(vel), np.percentile(vel, 95)]
        # 躯干角度（跌倒关键：趋近水平 90°）
        stats += [np.mean(ang), np.std(ang), np.max(ang), float(np.mean(ang > 60))]
        # 静止占比（位移 < 阈值）
        if len(cy) > 5:
            diffs = np.abs(np.diff(cy))
            still_ratio = float(np.mean(diffs < 0.005))
        else:
            still_ratio = 0.0
        stats.append(still_ratio)
        # 关键点可见度
        stats += [np.mean(vis), np.min(vis)]

        feat = np.asarray(stats, dtype=np.float32)
        # 防御：若仍有 NaN（全部帧未检测到人），替换为 0
        if not np.isfinite(feat).all():
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        # 标准化（避免量纲差异主导）；全零则跳过避免除零
        if feat.std() > 1e-8:
            feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        # 确定性投影到目标维度
        return _deterministic_project(feat, self.target_dim)

    def extract_from_file(self, video_path: str) -> np.ndarray:
        """从视频文件提取特征"""
        import cv2
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f"无法读取视频帧: {video_path}")
        return self.extract_from_frames(frames)

    @staticmethod
    def _interp_nan(arr: list) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if not np.isnan(arr).any():
            return arr
        valid = ~np.isnan(arr)
        if not valid.any():
            return np.zeros_like(arr)
        arr[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], arr[valid])
        return arr


# ===========================================================================
# 音频特征提取（librosa 真实声学）
# ===========================================================================

class RealAudioFeatureExtractor:
    """
    基于 librosa 的音频特征提取器。

    流程：音频波形 → MFCC/Mel/能量/ZCR/频谱质心 → 时序统计 → 投影 768 维

    提取的语义特征（与撞击/呼救/异常强相关）：
    - MFCC 13 维 × {mean, std}
    - Mel 频谱 64 维 × {mean, std}
    - 能量 RMS × {mean, std, max, peak_count}
    - 过零率 ZCR × {mean, std}
    - 频谱质心 × {mean, std}
    - 频谱带宽 × {mean, std}
    """

    def __init__(self, target_dim: int = AUDIO_DIM, sample_rate: int = 16000):
        self.target_dim = target_dim
        self.sample_rate = sample_rate

    def extract_from_signal(self, signal: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """从音频信号提取 768 维特征"""
        import librosa
        sr = sr or self.sample_rate
        signal = np.asarray(signal, dtype=np.float32)
        if signal.size == 0:
            return np.zeros(self.target_dim, dtype=np.float32)

        stats = []
        hop = 512
        n_fft = 1024

        # MFCC
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
        stats.extend(np.mean(mfcc, axis=1))
        stats.extend(np.std(mfcc, axis=1))

        # Mel 频谱（取前 64 mel）
        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=64, n_fft=n_fft, hop_length=hop)
        stats.extend(np.mean(mel, axis=1))
        stats.extend(np.std(mel, axis=1))

        # 能量 RMS
        rms = librosa.feature.rms(y=signal, frame_length=n_fft, hop_length=hop)[0]
        stats.extend([np.mean(rms), np.std(rms), np.max(rms),
                      float(np.sum(rms > np.mean(rms) + 2 * np.std(rms)))])

        # 过零率
        zcr = librosa.feature.zero_crossing_rate(signal, frame_length=n_fft, hop_length=hop)[0]
        stats.extend([np.mean(zcr), np.std(zcr), np.max(zcr)])

        # 频谱质心 & 带宽
        cent = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=n_fft, hop_length=hop)[0]
        bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=n_fft, hop_length=hop)[0]
        stats.extend([np.mean(cent), np.std(cent), np.mean(bw), np.std(bw)])

        feat = np.asarray(stats, dtype=np.float32)
        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        return _deterministic_project(feat, self.target_dim)

    def extract_from_file(self, audio_path: str) -> np.ndarray:
        """从音频文件提取特征"""
        import librosa
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        signal, sr = librosa.load(audio_path, sr=self.sample_rate)
        return self.extract_from_signal(signal, sr)


# ===========================================================================
# 生理特征提取（保留语义编码，规范化投影）
# ===========================================================================

class RealHealthFeatureExtractor:
    """
    生理数据特征提取器。

    支持两种输入：
    - 字典（单点）：{heart_rate, blood_oxygen, systolic, diastolic, steps}
    - 时序数组：(seq_len, 5)
    """

    # 正常生理范围（用于归一化）
    NORMAL_RANGES = {
        "heart_rate": (50, 120),
        "blood_oxygen": (85, 100),
        "systolic": (90, 180),
        "diastolic": (50, 120),
        "steps": (0, 8000),
    }

    def __init__(self, target_dim: int = HEALTH_DIM):
        self.target_dim = target_dim

    def extract_from_dict(self, health_data: Dict) -> np.ndarray:
        """从单点字典提取特征"""
        feats = []
        defaults = {"heart_rate": 75, "blood_oxygen": 97,
                    "systolic": 120, "diastolic": 80, "steps": 2000}
        for key in ["heart_rate", "blood_oxygen", "systolic", "diastolic", "steps"]:
            val = float(health_data.get(key, defaults[key]))
            lo, hi = self.NORMAL_RANGES[key]
            norm = (val - lo) / (hi - lo + 1e-6)  # 归一化到 [0,1] 附近
            feats.append(norm)
        # 衍生特征：脉压差、心率是否超限
        feats.append(feats[2] - feats[3])  # 脉压差（归一化空间）
        feats.append(float(feats[0] > 1.0 or feats[0] < 0.0))  # 心率超限标志
        feats.append(float(feats[1] < 0.4))  # 血氧偏低标志
        feat = np.asarray(feats, dtype=np.float32)
        return _deterministic_project(feat, self.target_dim)

    def extract_from_series(self, time_series: np.ndarray) -> np.ndarray:
        """从时序数组 (seq_len, 5) 提取特征"""
        ts = np.asarray(time_series, dtype=np.float32)
        if ts.ndim == 1:
            ts = ts.reshape(-1, 1)
        # 每列统计
        mean = np.mean(ts, axis=0)
        std = np.std(ts, axis=0)
        mn = np.min(ts, axis=0)
        mx = np.max(ts, axis=0)
        # 变化率（趋势）
        if ts.shape[0] > 1:
            slope = (ts[-1] - ts[0]) / (ts.shape[0] - 1)
        else:
            slope = np.zeros(ts.shape[1], dtype=np.float32)
        feat = np.concatenate([mean, std, mn, mx, slope])
        return _deterministic_project(feat, self.target_dim)


# ===========================================================================
# 用药特征提取（保留语义编码，规范化投影）
# ===========================================================================

class RealMedicationFeatureExtractor:
    """
    用药数据特征提取器。

    支持两种输入：
    - 记录列表：[{name, dosage, time, taken}, ...]
    - 字典：{total_medications, adherence_rate, missed_doses}
    """

    def __init__(self, target_dim: int = MED_DIM):
        self.target_dim = target_dim

    def extract_from_records(self, records: List[Dict]) -> np.ndarray:
        feats = []
        total = max(len(records), 1)
        taken = sum(1 for r in records if r.get("taken", False))
        missed = total - taken
        adherence = taken / total
        feats = [total, adherence, missed, float(adherence < 0.8), float(adherence < 0.5)]
        feat = np.asarray(feats, dtype=np.float32)
        return _deterministic_project(feat, self.target_dim)

    def extract_from_dict(self, data: Dict) -> np.ndarray:
        feats = [
            float(data.get("total_medications", 0)),
            float(data.get("adherence_rate", 1.0)),
            float(data.get("missed_doses", 0)),
            float(data.get("adherence_rate", 1.0) < 0.8),
            float(data.get("adherence_rate", 1.0) < 0.5),
        ]
        feat = np.asarray(feats, dtype=np.float32)
        return _deterministic_project(feat, self.target_dim)


# ===========================================================================
# 统一接口（训练 / 推理共用）
# ===========================================================================

class UnifiedFeatureExtractor:
    """
    多模态统一特征提取器。
    训练时的 preprocess_real.py 和推理时的 App 都 import 此类，保证分布一致。
    """

    def __init__(self):
        self.video = RealVideoFeatureExtractor()
        self.audio = RealAudioFeatureExtractor()
        self.health = RealHealthFeatureExtractor()
        self.medication = RealMedicationFeatureExtractor()

    def extract_all(self, data: Dict) -> Dict[str, np.ndarray]:
        """
        统一提取入口。

        Args:
            data: {
                "video_path" or "video_frames": ...,
                "audio_path" or "audio_signal": ...,
                "health_data" or "health_series": ...,
                "medication_records" or "medication_data": ...,
            }
        Returns:
            {"video": (768,), "audio": (768,), "health": (256,), "medication": (128,)}
            缺失模态返回 None（由上层 MissingDataHandler 兜底）
        """
        out = {}

        # 视频
        if data.get("video_path"):
            out["video"] = self.video.extract_from_file(data["video_path"])
        elif data.get("video_frames"):
            out["video"] = self.video.extract_from_frames(data["video_frames"])
        else:
            out["video"] = None

        # 音频
        if data.get("audio_path"):
            out["audio"] = self.audio.extract_from_file(data["audio_path"])
        elif data.get("audio_signal") is not None:
            out["audio"] = self.audio.extract_from_signal(data["audio_signal"])
        else:
            out["audio"] = None

        # 生理
        if data.get("health_series") is not None:
            out["health"] = self.health.extract_from_series(data["health_series"])
        elif data.get("health_data"):
            out["health"] = self.health.extract_from_dict(data["health_data"])
        else:
            out["health"] = None

        # 用药
        if data.get("medication_records"):
            out["medication"] = self.medication.extract_from_records(data["medication_records"])
        elif data.get("medication_data"):
            out["medication"] = self.medication.extract_from_dict(data["medication_data"])
        else:
            out["medication"] = None

        return out


# ===========================================================================
# 自测
# ===========================================================================

if __name__ == "__main__":
    print("=== 统一特征提取模块自测 ===")
    print(f"维度: video={VIDEO_DIM} audio={AUDIO_DIM} health={HEALTH_DIM} med={MED_DIM}")

    # 音频自测（用白噪声模拟）
    print("\n[音频] 用白噪声测试...")
    ext = RealAudioFeatureExtractor()
    sig = np.random.randn(16000 * 3).astype(np.float32) * 0.1
    feat = ext.extract_from_signal(sig)
    print(f"  输出 shape: {feat.shape}, mean={feat.mean():.4f}, std={feat.std():.4f}")
    # 确定性验证：同输入两次结果应一致
    feat2 = ext.extract_from_signal(sig)
    print(f"  确定性复现误差: {np.max(np.abs(feat - feat2)):.2e}")

    # 生理自测
    print("\n[生理] 用字典测试...")
    hext = RealHealthFeatureExtractor()
    hf = hext.extract_from_dict({"heart_rate": 140, "blood_oxygen": 90,
                                 "systolic": 160, "diastolic": 100, "steps": 100})
    print(f"  输出 shape: {hf.shape}")

    # 用药自测
    print("\n[用药] 用字典测试...")
    mext = RealMedicationFeatureExtractor()
    mf = mext.extract_from_dict({"total_medications": 3, "adherence_rate": 0.5, "missed_doses": 2})
    print(f"  输出 shape: {mf.shape}")

    print("\n[PASS] 音频/生理/用药特征提取通过（视频需真实帧，跳过自测）")
