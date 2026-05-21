# 智护家应用版 - 多模态健康监护系统

## 📋 项目简介

**智护家应用版**是智慧居家养老多模态监护系统的用户界面版本，集成了训练好的多模态融合模型，提供友好的Web界面进行风险评估。

### 核心功能

- 📹 **视频分析**: 上传视频检测跌倒、异常行为
- 🔊 **音频分析**: 上传音频检测呼救声、撞击声
- ❤️ **健康监测**: 输入生理数据进行异常检测
- 💊 **用药管理**: 记录用药情况，计算依从性
- 🔗 **多模态融合**: 综合所有数据进行风险评估

---

## 📁 项目结构

```
smart_elderly_care_app/
├── app.py                        # Streamlit 主应用
├── config.py                     # 配置文件
├── requirements.txt              # Python 依赖
├── inference/                    # 推理模块
│   ├── __init__.py
│   ├── predictor.py              # 风险预测器
│   ├── feature_extractor.py      # 特征提取
│   └── missing_handler.py        # 缺失数据处理
├── pretrained_models/            # 训练好的模型
│   └── fusion_model.pt           # 融合模型权重
└── defaults/                     # 默认特征文件
    ├── normal_video.npy
    ├── silent_audio.npy
    ├── healthy_baseline.npy
    └── no_medication.npy
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd smart_elderly_care_app
pip install -r requirements.txt
```

### 2. 放置模型文件

将训练好的模型文件复制到 `pretrained_models/` 目录：

```bash
# 如果模型在 smart_elderly_care_v2/checkpoints/
cp ../smart_elderly_care_v2/checkpoints/best_model.pt ./pretrained_models/fusion_model.pt
```

### 3. 生成默认特征文件

```bash
python -c "from inference.missing_handler import generate_default_files; generate_default_files()"
```

### 4. 运行应用

```bash
streamlit run app.py
```

应用将在浏览器中打开，默认地址：`http://localhost:8501`

---

## 📖 使用说明

### 风险评估页面

1. **上传视频**（可选）: 支持mp4、avi、mov格式
2. **上传音频**（可选）: 支持wav、mp3、ogg格式
3. **输入生理数据**: 心率、血氧、血压、步数
4. **输入用药信息**: 应服药数、已服药数
5. **点击"开始风险评估"**: 系统将综合分析并给出结果

### 缺失数据处理

系统支持部分数据缺失的情况：

| 缺失模态 | 处理方式 |
|---------|---------|
| 视频 | 使用"正常活动"默认特征 |
| 音频 | 使用"安静环境"默认特征 |
| 生理数据 | 使用健康基线特征 |
| 用药信息 | 使用零向量（无用药） |

---

## 🔧 模型更新

当训练了新模型后，只需替换 `pretrained_models/fusion_model.pt` 文件即可：

```bash
# 从训练仓库复制新模型
cp ../smart_elderly_care_v2/checkpoints/best_model.pt ./pretrained_models/fusion_model.pt
```

**无需修改任何代码**，系统会自动加载新模型。

---

## 📊 风险等级说明

| 等级 | 名称 | 说明 | 处理建议 |
|------|------|------|---------|
| 0 | 低风险 | 状态正常 | 记录到系统，持续观察 |
| 1 | 中风险 | 存在潜在风险 | 推送通知，建议关注 |
| 2 | 高风险 | 需要立即关注 | 立即报警，联系确认 |

---

## 🔌 API 接口

如需在其他程序中调用预测功能：

```python
from inference import RiskPredictor, MultiModalFeatureExtractor

# 初始化
predictor = RiskPredictor()
extractor = MultiModalFeatureExtractor()

# 提取特征
health_feature = extractor.health_extractor.extract_from_dict({
    "heart_rate": 75,
    "blood_oxygen": 97,
    "systolic": 120,
    "diastolic": 80,
    "steps": 3000
})

# 预测
result = predictor.predict(
    video_features=None,  # 或提取的视频特征
    audio_features=None,  # 或提取的音频特征
    health_features=health_feature,
    medication_features=None
)

print(f"风险等级: {result['risk_name_cn']}")
print(f"置信度: {result['confidence']:.2%}")
```

---

## ⚠️ 注意事项

1. **模型文件**: 确保 `pretrained_models/fusion_model.pt` 存在
2. **首次运行**: 会自动生成默认特征文件
3. **视频/音频处理**: 需要安装 opencv-python 和 librosa
4. **GPU支持**: 如果有CUDA，会自动使用GPU加速

---

## 📝 更新日志

### v2.0.0 (2026-05-21)
- 集成训练好的多模态融合模型
- 实现Streamlit Web界面
- 支持缺失数据处理

---

## 👥 作者

多模态课程大作业项目

---

## 📄 许可证

MIT License
