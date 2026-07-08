# 智护家 V2 - 多模态融合网络训练版本

## 📋 项目简介

**智护家 V2**是智慧居家养老多模态监护系统的升级版本，实现了**可训练的多模态融合神经网络**，使用 Cross-Attention 机制融合视频、音频、生理数据和用药信息，进行综合风险评估。

### V1 vs V2 对比

| 特性 | V1 (规则版本) | V2 (神经网络版本) |
|------|-------------|-----------------|
| 视频检测 | MediaPipe + 规则 | 3D CNN + Transformer |
| 音频检测 | librosa + 规则 | CNN + Transformer |
| 生理监测 | 阈值 + Isolation Forest | Transformer 编码器 |
| 融合方式 | **固定权重加权** | **Cross-Attention 学习融合** |
| 可训练 | ❌ 不可训练 | ✅ 端到端可训练 |
| 输出 | 规则判断结果 | 概率分布 + 置信度 |

---

## 🏗️ 系统架构

```
输入数据                    编码器                    融合网络
─────────────────────────────────────────────────────────────────
视频 [batch, 3, 16, 224, 224] ──→ Video Encoder ──→ [batch, 768] ──┐
音频 [batch, 1, 128, 100]──────→ Audio Encoder ──→ [batch, 768] ──┼──→ Cross-Attention
生理 [batch, 100, 5]───────────→ Health Encoder ──→ [batch, 256] ──┤    Fusion Network
用药 [batch, 10]───────────────→ Med Encoder────→ [batch, 128] ──┘         │
                                                                         ▼
                                                              ┌─────────────────┐
                                                              │   分类器│
                                                              └─────────────────┘
                                                                         │
                                                                         ▼
                                                              风险等级: 低/中/高
```

---

## 📁 项目结构

```
smart_elderly_care_v2/
├── config.py              # 配置文件
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── inference.py           # 推理脚本
├── requirements.txt       # 依赖包
├── data/
│   ├── prepare_data.py    # 数据准备脚本
│   ├── raw/               # 原始数据
│   └── processed/         # 处理后的数据
├── models/
│   ├── encoders.py        # 各模态编码器
│   ├── fusion_net.py      # 多模态融合网络
│   └── dataset.py         # 数据加载器
├── checkpoints/           # 模型检查点
├── logs/                  # 训练日志
└── notebooks/             # 实验笔记本
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n smart_care python=3.10
conda activate smart_care

# 安装依赖
cd smart_elderly_care_v2
pip install -r requirements.txt
```

### 2. 生成数据

```bash
# 生成模拟数据（5000个样本）
python data/prepare_data.py
```

### 3. 训练模型

```bash
# 使用默认配置训练
python train.py

# 自定义参数训练
python train.py --epochs 50 --batch_size 64 --lr 1e-4

# 使用轻量模型
python train.py --model lite
```

### 4. 评估模型

```bash
# 评估最佳模型
python evaluate.py --checkpoint checkpoints/best_model.pt

# 指定输出目录
python evaluate.py --checkpoint checkpoints/best_model.pt --output_dir ./eval_results
```

### 5. 推理预测

```bash
# 演示模式
python inference.py --demo

# 使用特征文件预测
python inference.py \
    --video path/to/video_features.npy \
    --audio path/to/audio_features.npy \
    --health path/to/health_features.npy \
    --medication path/to/medication_features.npy
```

---

## 📖 模块说明

### 1. 数据准备 (`data/prepare_data.py`)

生成模拟多模态数据集，包含：
- 视频特征 (768维)
- 音频特征 (768维)
- 生理特征 (256维)
- 用药特征 (128维)
- 风险标签 (0=低, 1=中, 2=高)

```python
from data.prepare_data import MultiModalDataGenerator

generator = MultiModalDataGenerator()
dataset = generator.generate_dataset(num_samples=5000)
```

### 2. 编码器 (`models/encoders.py`)

| 编码器 | 输入 | 输出 | 架构 |
|--------|------|------|------|
| VideoEncoder | [B, 3, 16, 224, 224] | [B, 768] | 3D CNN + Transformer |
| AudioEncoder | [B, 1, 128, T] | [B, 768] | CNN + Transformer |
| HealthEncoder | [B, 100, 5] | [B, 256] | Transformer |
| MedicationEncoder | [B, 10] | [B, 128] | Embedding |

### 3. 融合网络 (`models/fusion_net.py`)

**MultiModalFusionNet**: 完整版融合网络
- 特征投影层（对齐维度）
- Cross-Modal Attention（跨模态交互）
- 自注意力层
- 分类头

**MultiModalFusionNetLite**: 轻量版融合网络
- 简单 MLP 融合
- 参数量更少
- 训练更快

### 4. 训练器 (`train.py`)

完整训练流程：
- 数据加载
- 模型初始化
- 损失函数（带类别权重）
- 优化器（AdamW）
- 学习率调度（Warmup + Cosine）
- 混合精度训练
- 早停机制
- 模型保存

---

## ⚙️ 配置说明

主要配置项（`config.py`）：

```python
# 模型配置
MODEL_CONFIG = {
    "video_encoder": {"hidden_dim": 768},
    "audio_encoder": {"hidden_dim": 768},
    "health_encoder": {"hidden_dim": 256},
    "medication_encoder": {"embedding_dim": 128},
    "fusion": {"hidden_dim": 512, "num_heads": 8, "num_layers": 4},
    "classifier": {"num_classes": 3},
}

# 训练配置
TRAIN_CONFIG = {
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "early_stopping": 10,
    "class_weights": [1.0, 2.0, 3.0],  # 处理类别不平衡
}
```

---

## 📊 输出示例

### 训练输出

```
Epoch 10/100:
  Train Loss: 0.4521, Train Acc: 82.34%
  Val Loss: 0.3892, Val Acc: 85.67%
  LR: 0.000085
  保存最佳模型: checkpoints/best_model.pt
```

### 评估输出

```
============================================================
模型评估报告（真实数据，迭代3）
============================================================

训练数据: ESC-50 真实音频 + Pexels 真实视频 + 半合成生理/用药 (900样本)

Full 模型 (Cross-Modal Attention, 8.5M 参数):
  准确率 (Accuracy): 1.0000
  精确率 (Precision-Macro): 1.0000
  召回率 (Recall-Macro): 1.0000
  F1分数 (F1-Macro): 1.0000
  AUC (Macro): 1.0000
  混淆矩阵: [[29,0,0],[0,28,0],[0,0,33]]

Lite 模型 (MLP Concat, 3.7M 参数):
  准确率 (Accuracy): 0.9444
  F1分数 (F1-Macro): 0.9444

消融实验:
  A1 Full (Cross-Attn):  100.00% (基线)
  A2 Lite (Concat):       94.44% (-5.56%)
  B  mask video:          98.89% (-1.11%)
  B  mask audio:          98.89% (-1.11%)
  B  mask health:         67.78% (-32.22%)
  B  mask medication:     97.78% (-2.22%)

结论: 生理数据最重要 (-32.22%)，四模态全部参与决策
```

### 推理输出

```
样本 1 (ESC-50 crying_baby + 高风险生理):
  风险等级: 高风险
  置信度: 97.2%
  概率分布: 低风险=0.8%, 中风险=1.9%, 高风险=97.2%
  模态权重: 视频=25.0% 音频=25.0% 生理=25.1% 用药=25.0%
  处理建议: 立即报警，联系确认
```

---

## 🔧 在A100上训练

### 连接服务器

```bash
ssh -p 49532 root@connect.nma1.seetacloud.com
# 密码: 5OBBVA54f6Ki
```

### 上传代码

```bash
# 本地打包
tar -czvf smart_elderly_care_v2.tar.gz smart_elderly_care_v2

# 上传
scp -P 49532 smart_elderly_care_v2.tar.gz root@connect.nma1.seetacloud.com:~/autod1-tmp/
```

### 服务器端操作

```bash
# 解压
cd ~/autod1-tmp
tar -xzvf smart_elderly_care_v2.tar.gz

# 安装依赖
cd smart_elderly_care_v2
pip install -r requirements.txt

# 训练
python train.py --epochs 100 --batch_size 64
```

---

## 📈 性能优化建议

1. **数据增强**：添加视频/音频数据增强
2. **预训练模型**：使用 VideoMAE、AST 预训练权重
3. **类别平衡**：使用 Focal Loss 或过采样
4. **模型集成**：训练多个模型进行集成
5. **超参数搜索**：使用 Optuna 进行自动调参

---

## 📝 更新日志

### v3.1.0 (2026-07)
- **视频模态全面升级**: Pexels 真实视频 (112个mp4，3风险类别) 替代零向量
- **MediaPipe Tasks API**: 0.10.35 PoseLandmarker (替代已废弃的 mp.solutions.pose)
- **特征质量飞跃**: 视频特征非零率 7.6% → 100%
- **数据规模扩大**: 380 → 900 样本 (train 720 / val 90 / test 90)
- **性能提升**: Full 测试准确率 94.74% → 100.00%
- **消融突破**: 视频模态贡献 0% → -1.11%，四模态全部参与决策
- 模态重要性: health(-32.22%) > medication(-2.22%) > video/audio(-1.11%)

### v3.0.0 (2026-07)
- 使用真实公开数据集训练（ESC-50 音频）
- 完成消融实验（A/B/C 三组）
- 统一特征提取器（单一真源）
- App 集成可解释性展示 + 周报功能

### v2.0.0 (2026-05)
- 重构为可训练的神经网络架构
- 实现 Cross-Attention 多模态融合
- 添加完整训练/评估/推理流程
- 支持混合精度训练
- 添加早停机制

---

## 👥 作者

多模态课程大作业项目

---

## 📄 许可证

MIT License
