# 迭代3 技术报告 — App 集成、可解释性与周报功能

> 生成时间：2026-07-07
> 关联文档：迭代0_现状自检报告.md、迭代2_真实数据训练与消融报告.md

---

## 1. 本轮迭代目标

在迭代2完成真实数据训练（Full 94.74% / Lite 97.37%）的基础上，迭代3聚焦**应用层集成与可交付性**：

| 子任务 | 目标 | 状态 |
|--------|------|------|
| 3-1 端到端联调 | App 用新 best_model.pt 对真实 ESC-50 音频推理 | ✅ 6/6 正确 |
| 3-2 可解释性展示 | App 展示模态权重 + 跨模态注意力热力图 | ✅ 完成 |
| 3-3 周报功能 | 基于真实预测的 7 天结构化周报 | ✅ 完成 |
| 3-4 演示用例 | demo.py + 演示音频包 | ✅ 3/3 正确 |

---

## 2. 端到端推理验证（3-1）

### 2.1 测试设计

- 模型：`smart_elderly_care_app/pretrained_models/fusion_model.pt`（迭代2 Full 模型，8.5M 参数）
- 数据：ESC-50 真实音频 + 按风险等级构造的半合成生理/用药数据
- 测试脚本：`smart_elderly_care_app/test_inference.py`

### 2.2 测试结果

| 风险等级 | 音频类别 | 预测 | 置信度 | 结果 |
|----------|----------|------|--------|------|
| 低风险 | dog | 低风险 | 51.6% | ✅ |
| 低风险 | dog | 低风险 | 52.9% | ✅ |
| 中风险 | coughing | 中风险 | 68.5% | ✅ |
| 中风险 | coughing | 中风险 | 70.4% | ✅ |
| 高风险 | crying_baby | 高风险 | 76.5% | ✅ |
| 高风险 | glass_breaking | 高风险 | 77.5% | ✅ |

**准确率：6/6 = 100%**

关键观察：
- 置信度随风险等级递增（51% → 70% → 77%），符合预期：高风险场景的生理指标偏离正常范围更远，模型决策更确定。
- 视频模态缺失时由 `MissingDataHandler` 填充默认特征，不影响推理。

### 2.3 修复的问题

1. **`inference` 包名冲突**：V2 目录下有 `inference.py` 模块文件，与 App 的 `inference/` 包冲突。修复方式：测试脚本中 V2_ROOT 用 `append` 而非 `insert(0)`，让 App 的 `inference` 包优先加载。
2. **probabilities 类型不匹配**：predictor 返回 dict `{'low':..,'medium':..,'high':..}`，测试脚本误用 `enumerate` 遍历。修复为按键名索引。

---

## 3. 可解释性中间结果展示（3-2）

### 3.1 实现方案

修改 [`predictor.py`](smart_elderly_care_app/inference/predictor.py) 的 `predict()` 方法，调用模型时传入 `return_attention=True`，返回两类可解释性数据：

| 数据 | 来源 | 形状 | 含义 |
|------|------|------|------|
| `modality_weights` | `model.modality_weights` (softmax) | `[4]` | 模型学习的各模态全局贡献权重 |
| `attention_matrix` | Cross-Modal Attention 各层平均 | `[4,4]` | 跨模态注意力矩阵（query→key） |

### 3.2 App 展示

在 [`app.py`](smart_elderly_care_app/app.py) 的 `show_prediction_result()` 中新增"模态贡献分析"区块：

- **模态全局权重柱状图**（Plotly Bar）：4 个模态的 softmax 权重
- **跨模态注意力热力图**（Plotly Heatmap）：4×4 矩阵，行=Query 模态，列=Key 模态

### 3.3 技术细节

注意力权重原始形状为 `[batch, num_heads=8, 4, 4]`（多头注意力），需对 heads 维度求平均，再对多层（`cross_attention_layers`）求平均，得到 `[4,4]` 的可视化矩阵。

```python
for w in attn_weights_list:
    arr = w[0].detach().cpu().numpy()  # [num_heads, 4, 4]
    mats.append(arr.mean(axis=0))      # [4, 4]
attn_avg = np.stack(mats).mean(axis=0)  # [4, 4]
```

### 3.4 客观发现

当前模型（小数据集 304 训练样本）的 `modality_weights` 和注意力矩阵均接近均匀分布（~25%）。这说明：

- 模态权重参数尚未充分分化（初始化为 ones/4，训练数据不足以驱动明显偏移）。
- **消融实验（mask 法）比 modality_weights 更能反映真实模态重要性**：mask health 后准确率下降 18.42%，mask audio 下降 7.89%，这是行为层面的证据，比参数权重更可靠。

---

## 4. 健康周报功能（3-3）

### 4.1 模块设计

新建 [`inference/weekly_report.py`](smart_elderly_care_app/inference/weekly_report.py)，提供三个核心函数：

| 函数 | 功能 |
|------|------|
| `generate_weekly_records()` | 生成 N 天的评估记录（调用 RiskPredictor 真实预测） |
| `analyze_records()` | 统计风险分布、趋势、模态贡献、生成建议/提醒 |
| `generate_report_text()` | 生成纯文本周报 |

### 4.2 周报内容

一份完整周报包含：

1. **风险等级分布**：低/中/高各多少天，占比
2. **每日风险趋势**：日期 × 风险等级 × 置信度 × 场景描述
3. **健康指标周均值**：心率、血氧、血压、步数
4. **用药依从率**：周平均
5. **模态贡献**：模型权重周平均
6. **风险提醒**：基于阈值的自动告警（如"本周有1天高风险"）
7. **照护建议**：基于健康指标的个性化建议

### 4.3 App 集成

- sidebar 新增"📅 健康周报"导航
- `show_weekly_report()` 页面展示：概览卡片、风险趋势折线图、健康指标、模态贡献、提醒/建议、每日明细表格、完整文本、JSON 导出

### 4.4 测试结果

7 天模拟数据生成成功：

```
风险分布: 低=3天 中=3天 高=1天
平均置信度: 59.6%
用药依从率: 77.9%
健康周均: HR=96 SpO2=95% BP=141/90 steps=2114
风险提醒: 3 条
照护建议: 1 条
```

---

## 5. 演示用例（3-4）

### 5.1 演示脚本

[`demo.py`](smart_elderly_care_app/demo.py) 提供一键演示：

```bash
python demo.py              # 全部演示
python demo.py --scenarios  # 仅场景推理
python demo.py --weekly     # 仅周报
```

### 5.2 演示音频包

从 ESC-50 选取三个代表性音频复制到 `demo_audio/`：

| 文件 | 类别 | 风险等级 |
|------|------|----------|
| demo_low_risk_dog.wav | dog | 低风险 |
| demo_mid_risk_coughing.wav | coughing | 中风险 |
| demo_high_risk_crying.wav | crying_baby | 高风险 |

### 5.3 演示结果

三个场景全部预测正确（3/3），周报生成成功，报告保存至 `reports/demo_weekly_report.txt`。

---

## 6. 代码变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `smart_elderly_care_app/inference/predictor.py` | 修改 | predict() 返回 attention_weights + modality_weights |
| `smart_elderly_care_app/inference/weekly_report.py` | 新建 | 周报生成模块 |
| `smart_elderly_care_app/app.py` | 修改 | 新增模态贡献分析区块 + 周报页面 |
| `smart_elderly_care_app/test_inference.py` | 修改 | 修复包冲突 + probabilities 类型 + 注意力打印 |
| `smart_elderly_care_app/demo.py` | 新建 | 答辩演示脚本 |
| `smart_elderly_care_app/demo_audio/` | 新建 | 3 个演示音频 |
| `smart_elderly_care_app/pretrained_models/fusion_model.pt` | 更新 | 迭代2训练的 Full 模型 |

---

## 7. 当前系统整体状态

### 7.1 模型性能

| 模型 | 参数量 | 测试准确率 | F1-Macro | AUC |
|------|--------|-----------|----------|-----|
| Full (Cross-Modal Attention) | 8.5M | 94.74% | 0.9537 | 0.9925 |
| Lite (MLP Concat) | 3.7M | 97.37% | 0.9765 | - |

### 7.2 消融实验结论

| 实验 | 准确率 | 相对基线变化 |
|------|--------|-------------|
| A1 Full (Cross-Attn) | 94.74% | 基线 |
| A2 Lite (Concat) | 97.37% | +2.63% |
| B mask video | 94.74% | 0% |
| B mask audio | 86.84% | -7.89% |
| B mask health | 76.32% | -18.42% |
| B mask medication | 94.74% | 0% |

**关键结论**：生理数据是最重要模态（-18.42%），音频次之（-7.89%），视频和用药在本数据集上贡献为 0。

### 7.3 诚实分析

1. **Lite > Full**：小数据集（304 训练样本）下，简单模型泛化更好。Full 的 Cross-Attention 优势需更大数据量才能体现。
2. **视频模态无贡献**：URFD 未下载成功（rar 格式），视频特征为零向量，导致 video mask 无影响。
3. **注意力均匀分布**：modality_weights 和注意力矩阵均接近 25%，模型参数未充分分化。
4. **端到端推理可靠**：尽管参数未分化，模型在真实音频上的推理准确率达 100%（6/6），说明分类器学到了有效的决策边界。
