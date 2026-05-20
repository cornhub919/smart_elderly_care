"""
智护家 - 智慧居家养老多模态监护系统
配置文件
"""

# 系统配置
SYSTEM_CONFIG = {
    "project_name": "智护家",
    "version": "1.0.0",
    "description": "面向独居老人的多模态居家健康监护与陪护建议系统"
}

# 视频分析配置
VIDEO_CONFIG = {
    "frame_skip": 5,  # 每隔5帧处理一次
    "fall_threshold_height": 0.3,  # 重心高度下降阈值（相对于画面高度）
    "fall_threshold_time": 1.0,  # 跌倒判定时间（秒）
    "stillness_threshold": 60,  # 静止判定时间（秒）
    "pose_model": "mediapipe",  # 姿态估计模型
}

# 音频分析配置
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "frame_length": 1024,
    "hop_length": 512,
    "n_mels": 128,
    "help_keywords": ["救命", "帮帮我", "我摔倒了", "难受", "过来一下", "疼"],
    "impact_labels": ["impact", "crash", "bang", "fall", "glass"],
}

# 生理数据配置
HEALTH_CONFIG = {
    "heart_rate": {
        "low_threshold": 50,  # 心率过低阈值
        "high_threshold": 120,  # 心率过高阈值
    },
    "blood_oxygen": {
        "low_threshold": 90,  # 血氧过低阈值
    },
    "blood_pressure": {
        "systolic_low": 90,  # 收缩压过低
        "systolic_high": 160,  # 收缩压过高
        "diastolic_low": 60,  # 舒张压过低
        "diastolic_high": 100,  # 舒张压过高
    },
    "activity": {
        "low_steps_threshold": 500,  # 每日步数过低阈值
    },
    "sleep": {
        "min_hours": 4,  # 最少睡眠时长
        "max_hours": 12,  # 最多睡眠时长
    }
}

# 用药提醒配置
MEDICATION_CONFIG = {
    "reminder_advance_minutes": 5,  # 提前提醒时间
    "missed_threshold_minutes": 30,  # 漏服判定时间
}

# 多模态融合配置
FUSION_CONFIG = {
    "weights": {
        "video": 0.4,
        "audio": 0.25,
        "health": 0.25,
        "medication": 0.1,
    },
    "risk_levels": {
        "low": {"min": 0, "max": 0.4},
        "medium": {"min": 0.4, "max": 0.7},
        "high": {"min": 0.7, "max": 1.0},
    }
}

# 报警配置
ALARM_CONFIG = {
    "low_risk": "记录到系统，写入周报",
    "medium_risk": "推送子女，建议关注",
    "high_risk": "立即报警，通知子女或照护人员",
}
