"""
健康周报生成模块
================
基于一周内每日的多模态风险评估记录，生成结构化周报：
  - 风险等级分布统计
  - 模态贡献趋势
  - 异常事件汇总
  - 个性化建议

设计原则：不依赖外部库（仅 numpy），可直接被 app.py 调用。
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np


RISK_NAMES = {0: "低风险", 1: "中风险", 2: "高风险"}
RISK_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}


def generate_weekly_records(
    predictor,
    feature_extractor,
    days: int = 7,
    seed: int = 42,
) -> List[Dict]:
    """
    生成最近 N 天的模拟评估记录（用于演示周报功能）。

    每天根据不同的健康/音频场景调用 RiskPredictor 获得真实预测结果，
    记录风险等级、置信度、模态权重等。

    Args:
        predictor: RiskPredictor 实例
        feature_extractor: MultiModalFeatureExtractor 实例
        days: 生成天数
        seed: 随机种子（保证可复现）

    Returns:
        记录列表，每条含 date, risk_level, confidence, probabilities, modality_weights, scenario
    """
    rng = np.random.RandomState(seed)
    today = datetime.now()
    records = []

    # 每天的场景模板：(健康数据, 用药数据, 场景描述, 音频可选)
    # 音频用 None（缺失处理），主要靠健康+用药数据变化驱动预测
    scenarios = [
        (  # 正常日
            {"heart_rate": 72, "blood_oxygen": 98, "systolic": 120, "diastolic": 80, "steps": 3500},
            {"total_medications": 3, "adherence_rate": 0.95, "missed_doses": 0},
            "正常活动日",
        ),
        (  # 轻微不适
            {"heart_rate": 95, "blood_oxygen": 95, "systolic": 140, "diastolic": 92, "steps": 1800},
            {"total_medications": 3, "adherence_rate": 0.80, "missed_doses": 1},
            "活动量偏低",
        ),
        (  # 中度风险
            {"heart_rate": 110, "blood_oxygen": 93, "systolic": 150, "diastolic": 98, "steps": 1200},
            {"total_medications": 4, "adherence_rate": 0.65, "missed_doses": 2},
            "指标偏高+漏服",
        ),
        (  # 高风险
            {"heart_rate": 145, "blood_oxygen": 88, "systolic": 185, "diastolic": 115, "steps": 200},
            {"total_medications": 2, "adherence_rate": 0.40, "missed_doses": 4},
            "多项异常",
        ),
        (  # 恢复日
            {"heart_rate": 80, "blood_oxygen": 97, "systolic": 128, "diastolic": 82, "steps": 2800},
            {"total_medications": 3, "adherence_rate": 0.90, "missed_doses": 0},
            "恢复期",
        ),
    ]

    for i in range(days):
        date = (today - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        # 按天轮转场景，加入轻微随机扰动
        base_health, base_med, scenario_name = scenarios[i % len(scenarios)]
        health_data = {k: v + int(rng.randn() * 2) for k, v in base_health.items()}
        med_data = dict(base_med)

        try:
            health_feat = feature_extractor.health_extractor.extract_from_dict(health_data)
            med_feat = feature_extractor.medication_extractor.extract_from_dict(med_data)

            result = predictor.predict(
                video_features=None,
                audio_features=None,
                health_features=health_feat,
                medication_features=med_feat,
            )
            records.append({
                "date": date,
                "risk_level": result["risk_level"],
                "risk_name": result["risk_name_cn"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "modality_weights": result.get("modality_weights", {}),
                "missing_modalities": result.get("missing_modalities", []),
                "scenario": scenario_name,
                "health_data": health_data,
                "medication_data": med_data,
            })
        except Exception as e:
            records.append({
                "date": date,
                "risk_level": -1,
                "risk_name": "评估失败",
                "confidence": 0,
                "error": str(e),
                "scenario": scenario_name,
            })

    return records


def analyze_records(records: List[Dict]) -> Dict:
    """
    分析周记录，生成统计摘要。

    Returns:
        {
            "risk_distribution": {风险等级: 天数},
            "avg_confidence": float,
            "risk_trend": [每日风险等级],
            "high_risk_days": [...],
            "modality_weight_avg": {模态: 平均权重},
            "medication_adherence_avg": float,
            "health_avg": {指标: 平均值},
            "suggestions": [str],
            "alerts": [str],
        }
    """
    valid = [r for r in records if r.get("risk_level", -1) >= 0]
    if not valid:
        return {"error": "no valid records"}

    # 风险分布
    risk_dist = {0: 0, 1: 0, 2: 0}
    for r in valid:
        risk_dist[r["risk_level"]] = risk_dist.get(r["risk_level"], 0) + 1

    # 平均置信度
    avg_conf = float(np.mean([r["confidence"] for r in valid]))

    # 风险趋势
    risk_trend = [r["risk_level"] for r in valid]
    dates = [r["date"] for r in valid]

    # 高风险日
    high_risk_days = [
        {"date": r["date"], "scenario": r.get("scenario", ""), "confidence": r["confidence"]}
        for r in valid if r["risk_level"] >= 1
    ]

    # 模态权重平均
    mw_acc = {}
    mw_count = 0
    for r in valid:
        mw = r.get("modality_weights", {})
        if mw:
            mw_count += 1
            for k, v in mw.items():
                mw_acc[k] = mw_acc.get(k, 0) + v
    modality_avg = {k: v / mw_count for k, v in mw_acc.items()} if mw_count > 0 else {}

    # 健康指标平均
    health_acc = {}
    health_count = 0
    for r in valid:
        hd = r.get("health_data", {})
        if hd:
            health_count += 1
            for k, v in hd.items():
                health_acc[k] = health_acc.get(k, 0) + v
    health_avg = {k: v / health_count for k, v in health_acc.items()} if health_count > 0 else {}

    # 用药依从率平均
    med_adhs = []
    for r in valid:
        md = r.get("medication_data", {})
        if "adherence_rate" in md:
            med_adhs.append(md["adherence_rate"])
    med_avg = float(np.mean(med_adhs)) if med_adhs else 0

    # 生成建议
    suggestions = []
    alerts = []

    if risk_dist.get(2, 0) > 0:
        alerts.append(f"本周有 {risk_dist[2]} 天出现高风险，建议就医复查并加强监护。")
    if risk_dist.get(1, 0) >= 3:
        alerts.append(f"本周有 {risk_dist[1]} 天出现中风险，建议关注老人日常状态。")
    if risk_dist.get(0, 0) >= 5:
        suggestions.append("整体状态良好，继续保持当前的生活和用药习惯。")

    if med_avg < 0.7:
        suggestions.append(f"本周用药依从率仅 {med_avg:.0%}，建议设置服药提醒或家属协助。")
    elif med_avg >= 0.9:
        suggestions.append("用药依从性良好，请继续保持。")

    if health_avg.get("heart_rate", 80) > 100:
        suggestions.append(f"平均心率 {health_avg['heart_rate']:.0f} 偏高，建议监测心电图。")
    if health_avg.get("systolic", 120) > 140:
        alerts.append(f"平均收缩压 {health_avg['systolic']:.0f} 偏高，建议咨询医生。")
    if health_avg.get("blood_oxygen", 97) < 94:
        alerts.append(f"平均血氧 {health_avg['blood_oxygen']:.0f}% 偏低，建议检查呼吸功能。")
    if health_avg.get("steps", 3000) < 1500:
        suggestions.append("本周活动量偏低，建议适当增加散步等轻度运动。")

    if not suggestions:
        suggestions.append("各项指标平稳，请持续监测。")

    return {
        "risk_distribution": risk_dist,
        "avg_confidence": avg_conf,
        "risk_trend": risk_trend,
        "dates": dates,
        "high_risk_days": high_risk_days,
        "modality_weight_avg": modality_avg,
        "medication_adherence_avg": med_avg,
        "health_avg": health_avg,
        "suggestions": suggestions,
        "alerts": alerts,
        "total_days": len(valid),
    }


def generate_report_text(records: List[Dict], analysis: Dict) -> str:
    """生成纯文本周报（可用于导出/展示）"""
    lines = []
    lines.append("=" * 50)
    lines.append("           智护家 - 健康周报")
    lines.append("=" * 50)
    if records:
        lines.append(f"报告周期: {records[0]['date']} ~ {records[-1]['date']}")
    lines.append(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    rd = analysis.get("risk_distribution", {})
    lines.append("【风险等级分布】")
    for lv in [0, 1, 2]:
        cnt = rd.get(lv, 0)
        pct = cnt / max(analysis.get("total_days", 1), 1) * 100
        lines.append(f"  {RISK_NAMES[lv]}: {cnt} 天 ({pct:.0f}%)")
    lines.append(f"  平均置信度: {analysis.get('avg_confidence', 0):.1%}")
    lines.append("")

    lines.append("【每日风险趋势】")
    for r in records:
        if r.get("risk_level", -1) >= 0:
            lines.append(f"  {r['date']}  {r['risk_name']:6s}  置信度{r['confidence']:.0%}  ({r.get('scenario','')})")
        else:
            lines.append(f"  {r['date']}  评估失败")
    lines.append("")

    ha = analysis.get("health_avg", {})
    if ha:
        lines.append("【健康指标周均值】")
        lines.append(f"  心率: {ha.get('heart_rate',0):.0f} 次/分")
        lines.append(f"  血氧: {ha.get('blood_oxygen',0):.0f}%")
        lines.append(f"  收缩压: {ha.get('systolic',0):.0f} mmHg")
        lines.append(f"  舒张压: {ha.get('diastolic',0):.0f} mmHg")
        lines.append(f"  步数: {ha.get('steps',0):.0f} 步/天")
        lines.append("")

    lines.append(f"【用药依从率】{analysis.get('medication_adherence_avg',0):.0%}")
    lines.append("")

    mw = analysis.get("modality_weight_avg", {})
    if mw:
        lines.append("【模态贡献（模型权重）】")
        for k, v in mw.items():
            lines.append(f"  {k}: {v:.1%}")
        lines.append("")

    alerts = analysis.get("alerts", [])
    if alerts:
        lines.append("【风险提醒】")
        for a in alerts:
            lines.append(f"  ! {a}")
        lines.append("")

    suggestions = analysis.get("suggestions", [])
    if suggestions:
        lines.append("【照护建议】")
        for s in suggestions:
            lines.append(f"  - {s}")
        lines.append("")

    lines.append("=" * 50)
    return "\n".join(lines)


def save_report_json(records: List[Dict], analysis: Dict, text: str, path: str) -> None:
    """保存周报到 JSON 文件"""
    report_data = {
        "generated_at": datetime.now().isoformat(),
        "records": records,
        "analysis": analysis,
        "text": text,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
