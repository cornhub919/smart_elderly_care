"""
答辩演示脚本
============
一键展示智护家系统全部核心功能：
  1. 三个典型场景的端到端多模态推理（低/中/高风险）
  2. 模态贡献与注意力权重可视化数据
  3. 周报自动生成

使用方式：
  python demo.py              # 运行全部演示
  python demo.py --scenarios  # 仅场景推理
  python demo.py --weekly     # 仅周报

答辩录屏时建议按此顺序演示。
"""
import os
import sys
import argparse

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_ROOT)

from inference.predictor import RiskPredictor
from inference.feature_extractor import (
    AudioFeatureExtractor, HealthFeatureExtractor, MedicationFeatureExtractor,
    MultiModalFeatureExtractor,
)
from inference.weekly_report import (
    generate_weekly_records, analyze_records, generate_report_text,
)

RISK_NAME = {0: "低风险", 1: "中风险", 2: "高风险"}

# ============================================================
# 演示场景定义
# ============================================================
DEMO_SCENARIOS = [
    {
        "name": "场景1: 日常平静（低风险）",
        "audio": "demo_audio/demo_low_risk_dog.wav",
        "audio_desc": "狗叫声 (ESC-50 category=dog)",
        "health": {"heart_rate": 72, "blood_oxygen": 98, "systolic": 120, "diastolic": 80, "steps": 3500},
        "medication": {"total_medications": 3, "adherence_rate": 0.95, "missed_doses": 0},
        "expected": 0,
    },
    {
        "name": "场景2: 咳嗽+指标偏高（中风险）",
        "audio": "demo_audio/demo_mid_risk_coughing.wav",
        "audio_desc": "咳嗽声 (ESC-50 category=coughing)",
        "health": {"heart_rate": 110, "blood_oxygen": 93, "systolic": 150, "diastolic": 98, "steps": 1200},
        "medication": {"total_medications": 4, "adherence_rate": 0.65, "missed_doses": 2},
        "expected": 1,
    },
    {
        "name": "场景3: 婴儿啼哭+多项异常（高风险）",
        "audio": "demo_audio/demo_high_risk_crying.wav",
        "audio_desc": "婴儿啼哭 (ESC-50 category=crying_baby)",
        "health": {"heart_rate": 145, "blood_oxygen": 88, "systolic": 185, "diastolic": 115, "steps": 200},
        "medication": {"total_medications": 2, "adherence_rate": 0.40, "missed_doses": 4},
        "expected": 2,
    },
]


def print_sep(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def run_scenarios(predictor, fe):
    """运行三个演示场景"""
    ae = AudioFeatureExtractor()
    he = HealthFeatureExtractor()
    me = MedicationFeatureExtractor()

    print_sep("PART 1: 多模态风险推理演示")
    print(f"模型: MultiModalFusionNet (Cross-Modal Attention)")
    print(f"输入模态: 视频(缺失->默认) + 音频(ESC-50真实) + 生理 + 用药")

    correct = 0
    for scenario in DEMO_SCENARIOS:
        print_sep(scenario["name"])
        print(f"  音频: {scenario['audio_desc']}")
        print(f"  生理: HR={scenario['health']['heart_rate']} SpO2={scenario['health']['blood_oxygen']}% "
              f"BP={scenario['health']['systolic']}/{scenario['health']['diastolic']} "
              f"steps={scenario['health']['steps']}")
        print(f"  用药: {scenario['medication']['total_medications']}种 依从率={scenario['medication']['adherence_rate']:.0%} "
              f"漏服={scenario['medication']['missed_doses']}")

        audio_path = os.path.join(APP_ROOT, scenario["audio"])
        if not os.path.exists(audio_path):
            print(f"  [WARN] 音频文件不存在: {audio_path}, 跳过音频")
            audio_feat = None
        else:
            audio_feat = ae.extract_from_file(audio_path)

        health_feat = he.extract_from_dict(scenario["health"])
        med_feat = me.extract_from_dict(scenario["medication"])

        result = predictor.predict(
            video_features=None,
            audio_features=audio_feat,
            health_features=health_feat,
            medication_features=med_feat,
        )

        pred = result["risk_level"]
        match = "OK" if pred == scenario["expected"] else "MISS"
        if pred == scenario["expected"]:
            correct += 1

        probs = result.get("probabilities", {})
        prob_str = " ".join([f"{RISK_NAME[i]}={probs.get(k,0):.1%}" for i, k in enumerate(["low","medium","high"])])

        print(f"\n  >>> 预测结果: {result['risk_name_cn']} (置信度 {result['confidence']:.1%}) [{match}]")
        print(f"      期望: {RISK_NAME[scenario['expected']]}")
        print(f"      概率: {prob_str}")

        mw = result.get("modality_weights", {})
        if mw:
            mw_str = " ".join([f"{k}={v:.1%}" for k, v in mw.items()])
            print(f"      模态权重: {mw_str}")

        attn = result.get("attention_matrix")
        if attn:
            import numpy as np
            mat = np.array(attn)
            mnames = result.get("modality_names", ["V","A","H","M"])
            print(f"      注意力矩阵 (avg over heads/layers):")
            for i in range(4):
                r = "  ".join([f"{float(mat[i][j]):.2f}" for j in range(4)])
                print(f"        {mnames[i]:>4s}: {r}")

        if result.get("missing_modalities"):
            print(f"      缺失模态: {', '.join(result['missing_modalities'])}")

    print_sep("场景推理汇总")
    print(f"  {correct}/{len(DEMO_SCENARIOS)} 场景预测正确")
    return correct


def run_weekly(predictor, fe):
    """运行周报生成演示"""
    print_sep("PART 2: 健康周报自动生成演示")
    print("  模拟最近7天的多模态评估记录，生成结构化周报...")

    records = generate_weekly_records(predictor, fe, days=7, seed=42)
    analysis = analyze_records(records)

    rd = analysis.get("risk_distribution", {})
    print(f"\n  周报周期: {records[0]['date']} ~ {records[-1]['date']}")
    print(f"  风险分布: 低={rd.get(0,0)}天 中={rd.get(1,0)}天 高={rd.get(2,0)}天")
    print(f"  平均置信度: {analysis.get('avg_confidence',0):.1%}")
    print(f"  用药依从率: {analysis.get('medication_adherence_avg',0):.1%}")

    ha = analysis.get("health_avg", {})
    if ha:
        print(f"  健康周均: HR={ha.get('heart_rate',0):.0f} SpO2={ha.get('blood_oxygen',0):.0f}% "
              f"BP={ha.get('systolic',0):.0f}/{ha.get('diastolic',0):.0f} steps={ha.get('steps',0):.0f}")

    alerts = analysis.get("alerts", [])
    if alerts:
        print(f"\n  风险提醒 ({len(alerts)} 条):")
        for a in alerts:
            print(f"    ! {a}")

    suggestions = analysis.get("suggestions", [])
    if suggestions:
        print(f"\n  照护建议 ({len(suggestions)} 条):")
        for s in suggestions:
            print(f"    - {s}")

    text = generate_report_text(records, analysis)
    report_path = os.path.join(APP_ROOT, "reports", "demo_weekly_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  完整周报已保存至: {report_path}")
    print_sep("周报生成完成")


def main():
    parser = argparse.ArgumentParser(description="智护家答辩演示脚本")
    parser.add_argument("--scenarios", action="store_true", help="仅运行场景推理")
    parser.add_argument("--weekly", action="store_true", help="仅运行周报生成")
    args = parser.parse_args()

    print_sep("智护家 - 多模态健康监护系统 答辩演示")
    print("  模型: MultiModalFusionNet (Cross-Modal Attention)")
    print("  训练数据: ESC-50 真实音频 + URFD + 半合成生理/用药")
    print("  测试准确率: 94.74% (Full) / 97.37% (Lite)")

    predictor = RiskPredictor()
    fe = MultiModalFeatureExtractor()

    run_s = True
    run_w = True
    if args.scenarios or args.weekly:
        run_s = args.scenarios
        run_w = args.weekly

    if run_s:
        run_scenarios(predictor, fe)
    if run_w:
        run_weekly(predictor, fe)

    print_sep("演示完成")


if __name__ == "__main__":
    main()
