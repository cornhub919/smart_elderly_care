"""
智护家应用版 - 配置文件
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型配置（与训练时保持一致）
MODEL_CONFIG = {
    "video_encoder": {
        "hidden_dim": 768,
    },
    "audio_encoder": {
        "hidden_dim": 768,
    },
    "health_encoder": {
        "hidden_dim": 256,
    },
    "medication_encoder": {
        "embedding_dim": 128,
    },
    "fusion": {
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 4,
        "dropout": 0.2,
    },
    "classifier": {
        "num_classes": 3,
        "hidden_dims": [256, 128],
        "dropout": 0.3,
    }
}

# 风险等级定义
RISK_LEVELS = {
    0: {
        "name": "low",
        "name_cn": "低风险",
        "color": "#4CAF50",
        "description": "记录到系统，写入周报，持续观察。",
        "action": "无需立即行动"
    },
    1: {
        "name": "medium", 
        "name_cn": "中风险",
        "color": "#FF9800",
        "description": "推送通知给子女，建议关注老人状态。",
        "action": "建议电话确认"
    },
    2: {
        "name": "high",
        "name_cn": "高风险",
        "color": "#F44336",
        "description": "立即报警！通知子女或照护人员，必要时联系急救。",
        "action": "立即联系老人"
    }
}

# 特征维度
FEATURE_DIMS = {
    "video": 768,
    "audio": 768,
    "health": 256,
    "medication": 128,
}

# 默认特征文件路径
DEFAULT_FEATURES = {
    "video": os.path.join(PROJECT_ROOT, "defaults", "normal_video.npy"),
    "audio": os.path.join(PROJECT_ROOT, "defaults", "silent_audio.npy"),
    "health": os.path.join(PROJECT_ROOT, "defaults", "healthy_baseline.npy"),
    "medication": os.path.join(PROJECT_ROOT, "defaults", "no_medication.npy"),
}

# 模型路径
MODEL_PATH = os.path.join(PROJECT_ROOT, "pretrained_models", "fusion_model.pt")

# 生理数据阈值
HEALTH_THRESHOLDS = {
    "heart_rate": {"low": 50, "high": 120, "unit": "bpm"},
    "blood_oxygen": {"low": 90, "high": 100, "unit": "%"},
    "systolic": {"low": 90, "high": 160, "unit": "mmHg"},
    "diastolic": {"low": 60, "high": 100, "unit": "mmHg"},
}

# 用药配置
MEDICATION_CONFIG = {
    "common_medications": [
        "降压药", "降糖药", "心脏病药", "止痛药", 
        "安眠药", "抗凝药", "钙片", "维生素"
    ],
    "default_dosage": "1片",
}
