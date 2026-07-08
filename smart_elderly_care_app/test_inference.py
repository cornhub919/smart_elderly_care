"""
App 端到端推理测试
==================
用真实 ESC-50 音频 + 半合成健康/用药数据，验证 App 完整推理链路：
  音频文件 -> AudioFeatureExtractor -> 特征向量 -> RiskPredictor -> 风险等级
"""
import os
import sys
import csv
import numpy as np

# 确保能 import App 模块
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_ROOT)

# V2_ROOT 仅用于定位 ESC-50 数据路径；不 insert 到 sys.path 前端，
# 因为 V2 下有 inference.py 模块，会与 App 的 inference/ 包冲突。
# feature_extractor.py 内部已自行 append V2_ROOT 以导入统一提取器。
V2_ROOT = os.path.join(os.path.dirname(APP_ROOT), "smart_elderly_care_v2")
if V2_ROOT not in sys.path:
    sys.path.append(V2_ROOT)

from inference.feature_extractor import (
    AudioFeatureExtractor,
    HealthFeatureExtractor,
    MedicationFeatureExtractor,
)
from inference.predictor import RiskPredictor


# ESC-50 类别 -> 风险等级
ESC50_RISK_MAP = {
    "crying_baby": 2, "screaming": 2, "glass_breaking": 2,
    "sneezing": 1, "coughing": 1, "snoring": 1,
    "rain": 0, "wind": 0, "crickets": 0, "water_drops": 0,
    "dog": 0, "rooster": 0,
}

RISK_NAME = {0: "低风险", 1: "中风险", 2: "高风险"}


def find_test_audio(esc50_meta, esc50_audio, risk, n=2):
    """从 ESC-50 meta 找指定风险等级的音频文件"""
    results = []
    with open(esc50_meta, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row.get("category", "")
            if ESC50_RISK_MAP.get(cat) == risk:
                fpath = os.path.join(esc50_audio, row["filename"])
                if os.path.exists(fpath):
                    results.append((row["filename"], cat))
                    if len(results) >= n:
                        break
    return results


def health_by_risk(risk):
    """按风险等级生成健康数据"""
    if risk == 0:
        return {"heart_rate": 72, "blood_oxygen": 98, "systolic": 120, "diastolic": 80, "steps": 3500}
    elif risk == 1:
        return {"heart_rate": 110, "blood_oxygen": 93, "systolic": 150, "diastolic": 98, "steps": 1200}
    else:
        return {"heart_rate": 145, "blood_oxygen": 88, "systolic": 185, "diastolic": 115, "steps": 200}


def med_by_risk(risk):
    """按风险等级生成用药数据"""
    if risk == 0:
        return {"total_medications": 3, "adherence_rate": 0.95, "missed_doses": 0}
    elif risk == 1:
        return {"total_medications": 4, "adherence_rate": 0.65, "missed_doses": 2}
    else:
        return {"total_medications": 2, "adherence_rate": 0.4, "missed_doses": 4}


def main():
    esc50_audio = os.path.join(V2_ROOT, "data", "raw", "esc50", "ESC-50-master", "audio")
    esc50_meta = os.path.join(V2_ROOT, "data", "raw", "esc50", "ESC-50-master", "meta", "esc50.csv")

    if not os.path.exists(esc50_meta):
        print("[ERROR] ESC-50 meta.csv not found at:", esc50_meta)
        return

    print("=" * 60)
    print("[TEST] App End-to-End Inference with Real Audio")
    print("=" * 60)

    # 加载预测器
    predictor = RiskPredictor()
    ae = AudioFeatureExtractor()
    he = HealthFeatureExtractor()
    me = MedicationFeatureExtractor()

    # 测试三个风险等级
    correct = 0
    total = 0

    for risk in [0, 1, 2]:
        print(f"\n--- Testing Risk Level {risk} ({RISK_NAME[risk]}) ---")
        audio_files = find_test_audio(esc50_meta, esc50_audio, risk, n=2)

        for filename, category in audio_files:
            audio_path = os.path.join(esc50_audio, filename)
            try:
                audio_feat = ae.extract_from_file(audio_path)
                health_feat = he.extract_from_dict(health_by_risk(risk))
                med_feat = me.extract_from_dict(med_by_risk(risk))

                result = predictor.predict(
                    video_features=None,
                    audio_features=audio_feat,
                    health_features=health_feat,
                    medication_features=med_feat,
                )

                pred_risk = result["risk_level"]
                pred_name = result["risk_name_cn"]
                probs = result.get("probabilities", {})
                # probabilities 是 dict: {'low':..,'medium':..,'high':..}
                prob_keys = ["low", "medium", "high"]
                prob_str = " ".join(
                    [f"{RISK_NAME[i]}={probs.get(k, 0):.1%}" for i, k in enumerate(prob_keys)]
                )

                match = "OK" if pred_risk == risk else "MISS"
                if pred_risk == risk:
                    correct += 1
                total += 1

                print(f"  [{match}] {filename} (cat={category})")
                print(f"        expected={RISK_NAME[risk]}  predicted={pred_name}")
                print(f"        probs: {prob_str}")
                print(f"        confidence: {result.get('confidence', 0):.1%}")
                # 模态权重
                mw = result.get("modality_weights", {})
                mw_str = " ".join([f"{k}={v:.1%}" for k, v in mw.items()])
                print(f"        modality_weights: {mw_str}")
                # 注意力矩阵第一行
                attn = result.get("attention_matrix")
                if attn:
                    import numpy as _np
                    mat = _np.array(attn)
                    mnames = result.get("modality_names", ["video", "audio", "health", "med"])
                    row0 = " ".join([f"{mnames[j]}={float(mat[0][j]):.2f}" for j in range(4)])
                    print(f"        attn[video->*]: {row0}")
                    # 打印完整矩阵便于观察
                    for i in range(4):
                        r = " ".join([f"{float(mat[i][j]):.2f}" for j in range(4)])
                        print(f"        attn[{mnames[i]}->*]: {r}")
            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")

    print("\n" + "=" * 60)
    if total > 0:
        print(f"Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
    else:
        print("Results: 0/0 (no valid test cases)")
    print("=" * 60)


if __name__ == "__main__":
    main()
