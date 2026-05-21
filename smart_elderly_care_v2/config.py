"""
智护家 V2 - 配置文件
多模态融合网络训练版本
"""

from omegaconf import DictConfig, OmegaConf
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据配置
DATA_CONFIG = {
    # 数据路径
    "raw_data_dir": os.path.join(PROJECT_ROOT, "data/raw"),
    "processed_data_dir": os.path.join(PROJECT_ROOT, "data/processed"),
    "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
    "log_dir": os.path.join(PROJECT_ROOT, "logs"),
    
    # 数据集配置
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    
    # 多模态数据配置
    "video": {
        "num_frames": 16,          # 每个视频片段的帧数
        "frame_size": 224,         # 帧大小
        "fps": 30,                 # 原始视频帧率
    },
    "audio": {
        "sample_rate": 16000,      # 采样率
        "duration": 5.0,           # 音频片段时长（秒）
        "n_mels": 128,             # Mel频谱通道数
    },
    "health": {
        "sequence_length": 100,    # 时间序列长度
        "features": ["heart_rate", "blood_oxygen", "systolic", "diastolic", "steps"],
    },
    "medication": {
        "embedding_dim": 128,
    }
}

# 模型配置
MODEL_CONFIG = {
    # 视频编码器
    "video_encoder": {
        "name": "video_mae",       # video_mae / timesformer / slowfast
        "pretrained": True,
        "hidden_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "freeze_backbone": False,  # 是否冻结骨干网络
    },
    
    # 音频编码器
    "audio_encoder": {
        "name": "ast",             # ast / panns / beats
        "pretrained": True,
        "hidden_dim": 768,
        "freeze_backbone": False,
    },
    
    # 生理数据编码器
    "health_encoder": {
        "name": "transformer",
        "input_dim": 5,            # 输入特征维度
        "hidden_dim": 256,
        "num_layers": 4,
        "num_heads": 8,
        "dropout": 0.1,
    },
    
    # 用药编码器
    "medication_encoder": {
        "name": "embedding",
        "num_medications": 100,    # 药品种类数
        "embedding_dim": 128,
    },
    
    # 多模态融合网络
    "fusion": {
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 4,
        "dropout": 0.2,
        "fusion_type": "cross_attention",  # cross_attention / transformer / mlp
    },
    
    # 分类头
    "classifier": {
        "num_classes": 3,          # 低/中/高风险
        "hidden_dims": [256, 128],
        "dropout": 0.3,
    }
}

# 训练配置
TRAIN_CONFIG = {
    # 基本训练参数
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "gradient_clip": 1.0,
    
    # 学习率调度
    "scheduler": "cosine",         # cosine / linear / constant
    "min_lr": 1e-6,
    
    # 优化器
    "optimizer": "adamw",          # adam / adamw / sgd
    
    # 混合精度训练
    "use_amp": True,
    
    # 数据加载
    "num_workers": 8,
    "pin_memory": True,
    
    # 保存和验证
    "save_every": 5,               # 每N个epoch保存一次
    "validate_every": 1,           # 每N个epoch验证一次
    "early_stopping": 10,          # 早停耐心值
    
    # 损失函数
    "loss": "cross_entropy",       # cross_entropy / focal / label_smoothing
    "label_smoothing": 0.1,
    
    # 类别权重（处理不平衡数据）
    "class_weights": [1.0, 2.0, 3.0],  # 低/中/高风险的权重
}

# 评估配置
EVAL_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "auc"],
    "confusion_matrix": True,
    "per_class_metrics": True,
}

# 风险等级定义
RISK_LEVELS = {
    0: {"name": "low", "name_cn": "低风险", "description": "记录到系统，写入周报"},
    1: {"name": "medium", "name_cn": "中风险", "description": "推送子女，建议关注"},
    2: {"name": "high", "name_cn": "高风险", "description": "立即报警，通知子女"},
}

# 日志配置
LOG_CONFIG = {
    "use_wandb": False,            # 是否使用 Weights & Biases
    "wandb_project": "smart_elderly_care",
    "use_tensorboard": True,
    "log_level": "INFO",
}


def get_config():
    """获取完整配置"""
    config = {
        "data": DATA_CONFIG,
        "model": MODEL_CONFIG,
        "train": TRAIN_CONFIG,
        "eval": EVAL_CONFIG,
        "risk_levels": RISK_LEVELS,
        "log": LOG_CONFIG,
    }
    return OmegaConf.create(config)


if __name__ == "__main__":
    config = get_config()
    print(OmegaConf.to_yaml(config))
