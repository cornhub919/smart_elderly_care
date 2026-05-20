# 智护家 - 智慧居家养老多模态监护系统

## 📋 项目简介

**智护家**是一个面向独居老人的多模态居家健康监护与陪护建议系统。系统融合视频、音频、生理数据和用药信息，实现居家老人安全事件检测、健康异常预警、用药提醒和健康报告生成。

### 🎯 核心功能

| 功能模块 | 描述 |
|---------|------|
| 📹 **视频监测** | 跌倒检测、异常行为识别、长时间静止检测 |
| 🔊 **音频监测** | 呼救声识别、撞击声检测、异常静默检测 |
| ❤️ **健康监测** | 心率/血氧/血压异常检测、个性化健康基线 |
| 💊 **用药管理** | 服药提醒、漏服检测、用药依从性统计 |
| 🔗 **多模态融合** | 综合风险评估、三级报警机制 |
| 📊 **健康周报** | 自动生成健康报告和陪护建议 |

---

## 🏗️ 系统架构

```
数据采集层
    │
    ├── 摄像头：视频/图像
    ├── 麦克风：环境音频
    ├── 手环：心率、血氧、血压、运动量
    └── 用药计划：药品、剂量、时间、服药记录
            │
            ▼
数据预处理层
    │
    ├── 视频抽帧、人体检测、姿态估计
    ├── 音频降噪、特征提取
    ├── 生理数据清洗、缺失值处理
    └── 用药计划结构化
            │
            ▼
单模态智能分析层
    │
    ├── 视频行为识别模型
    ├── 音频异常事件识别模型
    ├── 生理指标异常检测模型
    └── 用药提醒与依从性分析模块
            │
            ▼
多模态融合决策层
    │
    ├── 跌倒风险融合判断
    ├── 健康异常风险评分
    ├── 异常行为综合判断
    └── 报警等级划分
            │
            ▼
应用服务层
    │
    ├── 实时报警
    ├── 子女端 App / Web 看板
    ├── 老人端语音提醒
    ├── 健康周报生成
    └── 陪护建议生成
```

---

## 📁 项目结构

```
smart_elderly_care/
├── app.py                      # Streamlit 主应用
├── requirements.txt            # Python 依赖
├── config/
│   └── config.py              # 系统配置
├── modules/
│   ├── __init__.py
│   ├── video/
│   │   ├── __init__.py
│   │   └── fall_detector.py   # 跌倒检测模块
│   ├── audio/
│   │   ├── __init__.py
│   │   └── audio_detector.py  # 音频异常检测模块
│   ├── health/
│   │   ├── __init__.py
│   │   └── health_monitor.py  # 生理数据监测模块
│   ├── medication/
│   │   ├── __init__.py
│   │   └── medication_manager.py  # 用药管理模块
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── fusion_engine.py   # 多模态融合模块
│   └── report/
│       ├── __init__.py
│       └── weekly_report.py   # 健康周报生成模块
├── data/
│   ├── sample_videos/         # 示例视频
│   ├── sample_audio/          # 示例音频
│   └── health_data/           # 健康数据
└── utils/                     # 工具函数
```

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- pip

### 2. 安装依赖

```bash
cd smart_elderly_care
pip install -r requirements.txt
```

### 3. 运行应用

```bash
streamlit run app.py
```

应用将在浏览器中打开，默认地址：`http://localhost:8501`

---

## 📖 模块说明

### 1. 视频跌倒检测模块 (`fall_detector.py`)

使用 MediaPipe Pose 进行人体姿态估计，结合规则判断检测跌倒。

**主要功能：**
- 人体关键点提取
- 重心高度计算
- 躯干角度计算
- 跌倒判断逻辑
- 静止状态检测

**使用示例：**
```python
from modules.video import FallDetector

detector = FallDetector()
is_fall, event, analysis = detector.detect_fall(frame)
```

### 2. 音频异常检测模块 (`audio_detector.py`)

检测呼救声、撞击声、摔倒声等异常音频事件。

**主要功能：**
- Mel频谱/MFCC特征提取
- 撞击声检测
- 尖叫声检测
- 咳嗽声检测
- 异常静默检测
- 关键词识别接口

**使用示例：**
```python
from modules.audio import AudioDetector

detector = AudioDetector()
results = detector.detect_from_file("audio.wav")
```

### 3. 生理数据监测模块 (`health_monitor.py`)

检测心率、血氧、血压等生理指标异常。

**主要功能：**
- 固定阈值检测
- 个性化基线建立
- Isolation Forest异常检测
- 健康趋势分析

**使用示例：**
```python
from modules.health import HealthMonitor

monitor = HealthMonitor()
monitor.set_baseline(historical_data)
anomalies = monitor.detect_anomaly(health_record)
```

### 4. 用药管理模块 (`medication_manager.py`)

管理用药计划、提醒服药、检测漏服。

**主要功能：**
- 添加/移除药品
- 今日用药计划
- 服药确认
- 漏服检测
- 用药依从性统计

**使用示例：**
```python
from modules.medication import MedicationManager

manager = MedicationManager()
manager.add_medication("降压药", "1片", 2, ["08:00", "20:00"])
reminders = manager.check_reminders()
```

### 5. 多模态融合模块 (`fusion_engine.py`)

融合视频、音频、生理数据、用药信息进行综合风险评估。

**主要功能：**
- 加权风险评分
- 三级风险等级划分
- 跌倒事件综合评估
- 主动确认机制

**使用示例：**
```python
from modules.fusion import FusionEngine

engine = FusionEngine()
assessment = engine.assess_fall_event(video_result, audio_result, health_result)
```

### 6. 健康周报生成模块 (`weekly_report.py`)

生成健康周报和陪护建议。

**主要功能：**
- 生理指标趋势分析
- 活动与睡眠统计
- 安全事件汇总
- 用药情况统计
- 陪护建议生成
- 风险提示生成

**使用示例：**
```python
from modules.report import ReportGenerator

generator = ReportGenerator()
report = generator.generate_weekly_report(health_data, safety_events, medication_stats)
text_report = generator.format_report_text(report)
```

---

## 🔧 配置说明

系统配置文件位于 `config/config.py`：

```python
# 视频分析配置
VIDEO_CONFIG = {
    "fall_threshold_height": 0.3,  # 重心高度下降阈值
    "fall_threshold_time": 1.0,    # 跌倒判定时间（秒）
    "stillness_threshold": 60,     # 静止判定时间（秒）
}

# 生理数据配置
HEALTH_CONFIG = {
    "heart_rate": {
        "low_threshold": 50,       # 心率过低阈值
        "high_threshold": 120,     # 心率过高阈值
    },
    "blood_oxygen": {
        "low_threshold": 90,       # 血氧过低阈值
    },
}

# 多模态融合配置
FUSION_CONFIG = {
    "weights": {
        "video": 0.4,
        "audio": 0.25,
        "health": 0.25,
        "medication": 0.1,
    },
}
```

---

## 📊 数据来源建议

### 视频跌倒数据集
- UR Fall Detection Dataset
- Le2i Fall Detection Dataset
- UP-Fall Detection Dataset

### 音频数据集
- ESC-50
- UrbanSound8K
- Google Speech Commands

### 生理数据
- 可使用模拟数据（已提供 `generate_sample_health_data()` 函数）
- 公开健康数据集

---

## 💡 创新点

1. **多模态融合**：融合视频、音频、生理数据、用药信息，提高异常识别可靠性
2. **个性化健康基线**：根据老人历史数据建立个人基线，实现个性化异常检测
3. **主动确认机制**：检测到异常时主动询问老人，减少误报
4. **从实时报警到长期陪护**：不仅关注突发事件，还关注长期健康趋势
5. **隐私友好设计**：视频本地处理，骨架点表示减少隐私泄露

---

## ⚠️ 注意事项

1. 本系统定位为**居家安全监测 + 健康风险预警 + 陪护辅助决策系统**，不替代医疗诊断
2. 具体阈值应支持个性化设置，不同老人身体情况不同
3. 报告中使用"风险提示""建议关注"等表达，避免写成"诊断结论"

---

## 📝 更新日志

### v1.0.0 (2024-01)
- 初始版本发布
- 实现六大核心模块
- 完成 Streamlit 演示界面

---

## 👥 作者

多模态课程大作业项目

---

## 📄 许可证

MIT License
