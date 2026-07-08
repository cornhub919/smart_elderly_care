"""用真实低风险视频特征均值重新生成 App 的 default 特征文件。
改版前 normal_video.npy 是随机噪声 *0.1，现在改为真实 daily 视频骨架特征均值。
healthy_baseline.npy 也用低风险 health 特征均值更新。
"""
import os
import sys
import numpy as np

V2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smart_elderly_care_v2")
V2_ROOT = os.path.abspath(V2_ROOT)
APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# 加载 v2 处理后的训练特征
train_dir = os.path.join(V2_ROOT, "data", "processed", "train")
video_feat = np.load(os.path.join(train_dir, "video_features.npy"))
audio_feat = np.load(os.path.join(train_dir, "audio_features.npy"))
health_feat = np.load(os.path.join(train_dir, "health_features.npy"))
med_feat = np.load(os.path.join(train_dir, "medication_features.npy"))
labels = np.load(os.path.join(train_dir, "labels.npy"))

# 低风险 (label=0) 样本的均值 = 正常活动基线
low_mask = labels == 0
print(f"低风险样本数: {low_mask.sum()}")

defaults_dir = os.path.join(APP_ROOT, "defaults")

# normal_video.npy: 低风险视频特征均值（真实日常活动骨架）
normal_video = video_feat[low_mask].mean(axis=0).astype(np.float32)
np.save(os.path.join(defaults_dir, "normal_video.npy"), normal_video)
print(f"normal_video.npy: shape={normal_video.shape}, norm={np.linalg.norm(normal_video):.4f}")

# silent_audio.npy: 低风险音频特征均值（正常背景声）
silent_audio = audio_feat[low_mask].mean(axis=0).astype(np.float32)
np.save(os.path.join(defaults_dir, "silent_audio.npy"), silent_audio)
print(f"silent_audio.npy: shape={silent_audio.shape}, norm={np.linalg.norm(silent_audio):.4f}")

# healthy_baseline.npy: 低风险健康特征均值
healthy = health_feat[low_mask].mean(axis=0).astype(np.float32)
np.save(os.path.join(defaults_dir, "healthy_baseline.npy"), healthy)
print(f"healthy_baseline.npy: shape={healthy.shape}, norm={np.linalg.norm(healthy):.4f}")

# no_medication.npy: 保持零向量（无用药记录）
no_med = np.zeros(med_feat.shape[1], dtype=np.float32)
np.save(os.path.join(defaults_dir, "no_medication.npy"), no_med)
print(f"no_medication.npy: shape={no_med.shape}, norm={np.linalg.norm(no_med):.4f}")

print("\n[OK] App default 特征文件已用真实低风险特征均值更新")
