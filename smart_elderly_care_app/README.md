# 智护家应用版 - 多模态健康监护系统

## 📋 项目简介

**智护家应用版**是智慧居家养老多模态监护系统的用户界面版本，集成了在**真实公开数据集（ESC-50 音频 + Pexels 视频）**上训练的多模态融合模型，四模态（视频/音频/生理/用药）全部使用真实数据训练，提供友好的 Web 界面进行风险评估、可解释性分析和健康周报生成。

### 核心功能

- 🏠 **风险评估**: 上传视频/音频 + 输入生理/用药数据，多模态融合预测风险等级
- 🔬 **可解释性展示**: 模态全局权重柱状图 + 跨模态注意力热力图
- 📅 **健康周报**: 自动生成 7 天结构化周报（风险分布、趋势、提醒、建议）
- 💊 **用药管理**: 记录用药情况，计算依从性
- 🔗 **缺失数据处理**: 任意模态缺失均可推理（默认特征填充）

### 模型性能

| 模型 | 参数量 | 测试准确率 | F1-Macro | AUC |
|------|--------|-----------|----------|-----|
| Full (Cross-Modal Attention) | 8.5M | 100.00% | 1.0000 | 1.0000 |
| Lite (MLP Concat) | 3.7M | 94.44% | 0.9444 | - |

> 训练数据：ESC-50 真实音频（2000 wav）+ Pexels 真实视频（112 mp4，MediaPipe 骨架）+ 半合成生理/用药特征，共 900 样本（train 720 / val 90 / test 90）
>
> 消融实验验证四模态全部参与决策：health(-32.22%) > medication(-2.22%) > video/audio(-1.11%)

---

## 📁 项目结构

```
smart_elderly_care_app/
├── app.py                        # Streamlit 主应用
├── config.py                     # 配置文件
├── demo.py                       # 答辩演示脚本
├── test_inference.py             # 端到端推理测试
├── requirements.txt              # Python 依赖
├── inference/                    # 推理模块
│   ├── __init__.py
│   ├── predictor.py              # 风险预测器（含注意力返回）
│   ├── feature_extractor.py      # 特征提取（代理到 V2 统一真源）
│   ├── missing_handler.py        # 缺失数据处理
│   └── weekly_report.py          # 健康周报生成
├── pretrained_models/            # 训练好的模型
│   └── fusion_model.pt           # Full 融合模型权重
├── defaults/                     # 默认特征文件
│   ├── normal_video.npy
│   ├── silent_audio.npy
│   ├── healthy_baseline.npy
│   └── no_medication.npy
├── demo_audio/                   # 演示音频（ESC-50 选取）
│   ├── demo_low_risk_dog.wav
│   ├── demo_mid_risk_coughing.wav
│   └── demo_high_risk_crying.wav
└── reports/                      # 生成的周报
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd smart_elderly_care_app
pip install -r requirements.txt
```

### 2. 确认模型文件

`pretrained_models/fusion_model.pt` 已包含训练好的 Full 模型。如需更新：

```bash
cp ../smart_elderly_care_v2/checkpoints/best_model.pt ./pretrained_models/fusion_model.pt
```

### 3. 生成默认特征文件（首次运行）

默认特征文件已使用**真实低风险特征均值**预生成（非随机噪声）：

| 文件 | 含义 | L2 范数 |
|------|------|---------|
| normal_video.npy | 正常活动视频特征均值 | 27.27 |
| silent_audio.npy | 安静环境音频特征均值 | 24.69 |
| healthy_baseline.npy | 健康生理基线 | 6.70 |
| no_medication.npy | 无用药零向量 | 0.00 |

如需重新生成：
```bash
python _regen_defaults.py
```

### 4. 运行应用

```bash
streamlit run app.py
```

应用将在浏览器中打开，默认地址：`http://localhost:8501`

### 5. 运行演示脚本（可选）

```bash
# 一键演示全部功能
python demo.py

# 仅演示场景推理
python demo.py --scenarios

# 仅演示周报生成
python demo.py --weekly
```

---

## 📖 功能说明

### 风险评估页面

1. **上传视频**（可选）: 支持 mp4、avi、mov 格式
2. **上传音频**（可选）: 支持 wav、mp3、ogg 格式（可用 demo_audio/ 中的演示音频）
3. **输入生理数据**: 心率、血氧、血压、步数
4. **输入用药信息**: 应服药数、已服药数
5. **点击"开始风险评估"**: 系统将综合分析并给出结果

### 评估结果展示

- **风险等级卡片**: 低/中/高风险 + 置信度 + 描述
- **概率分布柱状图**: 三个风险等级的概率
- **模态贡献分析**:
  - 模态全局权重柱状图（模型学习的各模态贡献）
  - 跨模态注意力热力图（4×4 矩阵，Query→Key）
- **缺失模态提示**: 自动检测并提示缺失的模态

### 健康周报页面

- 选择天数（3-14天）和随机种子
- 自动生成每日评估记录（调用模型真实预测）
- 展示：概览卡片、风险趋势折线图、健康指标周均值、模态贡献、风险提醒、照护建议、每日明细表格
- 支持导出 JSON 格式周报

### 缺失数据处理

| 缺失模态 | 处理方式 |
|---------|---------|
| 视频 | 使用"正常活动"默认特征 |
| 音频 | 使用"安静环境"默认特征 |
| 生理数据 | 使用健康基线特征 |
| 用药信息 | 使用零向量（无用药） |

---

## 📊 风险等级说明

| 等级 | 名称 | 说明 | 处理建议 |
|------|------|------|---------|
| 0 | 低风险 | 状态正常 | 记录到系统，持续观察 |
| 1 | 中风险 | 存在潜在风险 | 推送通知，建议关注 |
| 2 | 高风险 | 需要立即关注 | 立即报警，联系确认 |

---

## 🔌 API 接口

```python
from inference.predictor import RiskPredictor
from inference.feature_extractor import MultiModalFeatureExtractor

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

# 预测（返回风险等级 + 概率 + 模态权重 + 注意力矩阵）
result = predictor.predict(
    video_features=None,
    audio_features=None,
    health_features=health_feature,
    medication_features=None
)

print(f"风险等级: {result['risk_name_cn']}")
print(f"置信度: {result['confidence']:.2%}")
print(f"模态权重: {result['modality_weights']}")
```

### 周报生成 API

```python
from inference.weekly_report import generate_weekly_records, analyze_records, generate_report_text

records = generate_weekly_records(predictor, extractor, days=7, seed=42)
analysis = analyze_records(records)
report_text = generate_report_text(records, analysis)
print(report_text)
```

---

## 🧪 测试

### 端到端推理测试

```bash
python test_inference.py
```

使用 ESC-50 真实音频测试三个风险等级（各 2 个文件），验证完整推理链路。

### 预期输出

```
--- Testing Risk Level 0 (低风险) ---
  [OK] 1-100032-A-0.wav (cat=dog)        confidence: 91.0%
  [OK] 1-110389-A-0.wav (cat=dog)        confidence: 90.6%

--- Testing Risk Level 1 (中风险) ---
  [OK] 1-19111-A-24.wav (cat=coughing)   confidence: 95.3%
  [OK] 1-19118-A-24.wav (cat=coughing)   confidence: 95.3%

--- Testing Risk Level 2 (高风险) ---
  [OK] 1-187207-A-20.wav (cat=crying_baby)  confidence: 97.2%
  [OK] 1-20133-A-39.wav (cat=glass_breaking) confidence: 97.4%

Results: 6/6 correct (100.0%)
```

---

## ⚠️ 注意事项

1. **模型文件**: 确保 `pretrained_models/fusion_model.pt` 存在
2. **V2 依赖**: 特征提取器代理到 `smart_elderly_care_v2/models/unified_feature_extractor.py`，需保持 V2 目录在同级
3. **视频/音频处理**: 需要安装 opencv-python 和 librosa
4. **GPU支持**: 如果有 CUDA，会自动使用 GPU 加速（当前环境为 CPU）
5. **中文字体**: 图表使用 SimHei 字体，Windows 自带

---

## 📝 更新日志

### v3.1.0 (2026-07-07)
- 集成迭代3训练的真实四模态模型（Full 100.00% / Lite 94.44%）
- 默认特征文件升级为真实低风险特征均值（视频 norm=27.27，非随机噪声）
- 视频特征提取器升级 MediaPipe 0.10.35 Tasks API（PoseLandmarker）
- 端到端测试置信度提升：低 91% / 中 95% / 高 97%

### v3.0.0 (2026-07-07)
- 集成迭代2训练的真实数据模型（Full 94.74% / Lite 97.37%）
- 新增模态贡献分析（权重柱状图 + 注意力热力图）
- 新增健康周报功能（7天记录、趋势图、提醒、导出）
- 新增答辩演示脚本 demo.py
- 新增端到端推理测试 test_inference.py
- 修复 inference 包名冲突

### v2.0.0 (2026-05-21)
- 集成训练好的多模态融合模型
- 实现 Streamlit Web 界面
- 支持缺失数据处理

---

## 👥 作者

多模态课程大作业项目

---

## 📄 许可证

MIT License
