"""
真实数据预处理脚本
==================
将 Pexels 视频 + ESC-50 音频 通过统一特征提取器转为训练用 .npy。

输出格式与旧 prepare_data.py 完全兼容（dataset.py 可直接加载）：
  data/processed/{split}/{video,audio,health,medication,labels}_features.npy

样本构造逻辑
------------
1. 视频样本（来自 Pexels 视频集，MediaPipe PoseLandmarker 真实骨架提取）
   - falls/     → 标签 2 (高风险，跌倒动作)
   - struggle/  → 标签 1 (中风险，挣扎/困难起身)
   - daily/     → 标签 0 (低风险，日常活动)
   - 视频特征走 MediaPipe 真实提取（768 维投影）
   - 配对音频：跌倒配撞击/呼救声，日常配正常背景（从 ESC-50 采样）
   - 配对健康/用药：按风险等级用半合成场景规则生成

2. 纯音频样本（来自 ESC-50，补充样本量）
   - 呼救/尖叫 → 标签 2 (高风险，配合模拟跌倒)
   - 咳嗽/喷嚏 → 标签 1 (中风险，生理异常)
   - 雨/风/虫鸣 → 标签 0 (低风险，正常背景)
   - 视频特征配对同类真实视频特征（随机选取，非零向量）
   - 配对健康/用药按风险等级生成

3. 半合成补充样本（补充各类数量至 max_per_class）
   - 健康异常（心率偏高/血氧偏低）+ 用药不规律 → 按需补充
   - 视频/音频用同类真实特征 + 轻微扰动填充

这样既用真实特征，又保证三类样本数量均衡。

用法
----
    python data/preprocess_real.py                # 默认均衡生成
    python data/preprocess_real.py --max_per_class 200
    python data/preprocess_real.py --check_only   # 仅检查数据是否就绪
"""

import os
import sys
import csv
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V2_ROOT = os.path.dirname(SCRIPT_DIR)
if V2_ROOT not in sys.path:
    sys.path.append(V2_ROOT)

from config import get_config
from models.unified_feature_extractor import (
    UnifiedFeatureExtractor,
    RealVideoFeatureExtractor,
    RealAudioFeatureExtractor,
    RealHealthFeatureExtractor,
    RealMedicationFeatureExtractor,
    VIDEO_DIM, AUDIO_DIM, HEALTH_DIM, MED_DIM,
)

CFG = get_config()
RAW_DIR = CFG.data.raw_data_dir
PROCESSED_DIR = CFG.data.processed_data_dir

# ---------------------------------------------------------------------------
# 路径与数据集定位
# ---------------------------------------------------------------------------

ESC50_ROOT = os.path.join(RAW_DIR, "esc50", "ESC-50-master")
ESC50_AUDIO = os.path.join(ESC50_ROOT, "audio")
ESC50_META = os.path.join(ESC50_ROOT, "meta", "esc50.csv")

PEXELS_VIDEO_ROOT = os.path.join(RAW_DIR, "pexels_video")
# Pexels 视频集子目录 → 风险等级
PEXELS_CATEGORY_RISK = {
    "falls": 2,       # 跌倒动作 → 高风险
    "struggle": 1,    # 挣扎/困难起身 → 中风险
    "daily": 0,       # 日常活动 → 低风险
}

# ---------------------------------------------------------------------------
# ESC-50 类别 → 风险等级映射
# ---------------------------------------------------------------------------
# ESC-50 有 50 类，每类 40 条。选取与养老场景相关的类别。
ESC50_RISK_MAP = {
    # 高风险线索：呼救/尖叫/撞击
    "crying_baby": 2,      # 哭声（类比呼救）
    "screaming": 2,        # 尖叫
    "glass_breaking": 2,   # 玻璃碎裂（撞击）
    # 中风险：生理异常
    "sneezing": 1,
    "coughing": 1,
    "snoring": 1,
    # 低风险：正常背景
    "rain": 0,
    "wind": 0,
    "crickets": 0,
    "water_drops": 0,
    "dog": 0,
    "rooster": 0,
}


# ---------------------------------------------------------------------------
# 数据就绪检查
# ---------------------------------------------------------------------------

def check_datasets() -> dict:
    """检查各数据集就绪状态"""
    status = {}

    # ESC-50
    esc50_ready = (os.path.exists(ESC50_META)
                   and os.path.isdir(ESC50_AUDIO)
                   and len(list(Path(ESC50_AUDIO).glob("*.wav"))) > 100)
    status["esc50"] = esc50_ready
    if esc50_ready:
        status["esc50_wavs"] = len(list(Path(ESC50_AUDIO).glob("*.wav")))
    else:
        status["esc50_wavs"] = 0

    # Pexels 视频集：检测 falls/struggle/daily 三类子目录的 mp4 数量
    pexels_ready = False
    pexels_counts = {}
    if os.path.isdir(PEXELS_VIDEO_ROOT):
        for cat, _risk in PEXELS_CATEGORY_RISK.items():
            cat_dir = os.path.join(PEXELS_VIDEO_ROOT, cat)
            cnt = len(list(Path(cat_dir).glob("*.mp4"))) if os.path.isdir(cat_dir) else 0
            pexels_counts[cat] = cnt
        # 三类各至少 1 个视频即视为就绪
        pexels_ready = all(c > 0 for c in pexels_counts.values()) and len(pexels_counts) == 3
    status["pexels_video"] = pexels_ready
    status["pexels_video_counts"] = pexels_counts
    status["pexels_video_total"] = sum(pexels_counts.values()) if pexels_counts else 0

    # 兼容旧字段名（metadata 中曾用 urfd_ready）
    status["urfd"] = pexels_ready
    status["urfd_sequences"] = status["pexels_video_total"]

    return status


# ---------------------------------------------------------------------------
# ESC-50 加载
# ---------------------------------------------------------------------------

def load_esc50_metadata() -> list:
    """加载 ESC-50 元数据，返回 [(filename, category, target), ...]"""
    if not os.path.exists(ESC50_META):
        return []
    samples = []
    with open(ESC50_META, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row.get("category", "")
            if cat in ESC50_RISK_MAP:
                samples.append({
                    "filename": row["filename"],
                    "category": cat,
                    "target": int(row.get("target", 0)),
                    "risk": ESC50_RISK_MAP[cat],
                    "path": os.path.join(ESC50_AUDIO, row["filename"]),
                })
    return samples


# ---------------------------------------------------------------------------
# URFD 视频序列发现
# ---------------------------------------------------------------------------

def discover_pexels_videos() -> list:
    """
    发现 Pexels 视频集中的所有 mp4 文件。
    返回 [{"path": ..., "risk": 0/1/2, "name": ..., "category": ...}, ...]
    - falls/    → risk 2 (高风险)
    - struggle/ → risk 1 (中风险)
    - daily/    → risk 0 (低风险)
    """
    sequences = []
    if not os.path.isdir(PEXELS_VIDEO_ROOT):
        return sequences

    for cat, risk in PEXELS_CATEGORY_RISK.items():
        cat_dir = os.path.join(PEXELS_VIDEO_ROOT, cat)
        if not os.path.isdir(cat_dir):
            continue
        for mp4 in sorted(Path(cat_dir).glob("*.mp4")):
            sequences.append({
                "name": f"{cat}/{mp4.name}",
                "path": str(mp4),
                "risk": risk,
                "category": cat,
            })
    return sequences


# ---------------------------------------------------------------------------
# 半合成健康/用药特征生成（按风险等级，物理意义驱动）
# ---------------------------------------------------------------------------

class SemiSyntheticHealthMed:
    """
    按风险等级生成有物理意义的健康/用药特征。
    复用统一提取器的输入，保证特征分布与真实输入一致。
    """

    def __init__(self):
        self.health_ext = RealHealthFeatureExtractor()
        self.med_ext = RealMedicationFeatureExtractor()
        self.rng = random.Random(42)

    def generate(self, risk: int):
        """
        返回 (health_feat, med_feat)
        risk: 0 低 / 1 中 / 2 高
        """
        # 生理数据按风险等级
        if risk == 0:
            health_data = {
                "heart_rate": self.rng.uniform(60, 90),
                "blood_oxygen": self.rng.uniform(96, 100),
                "systolic": self.rng.uniform(110, 130),
                "diastolic": self.rng.uniform(70, 85),
                "steps": self.rng.uniform(2000, 6000),
            }
            med_data = {"total_medications": self.rng.randint(2, 4),
                        "adherence_rate": self.rng.uniform(0.9, 1.0),
                        "missed_doses": 0}
        elif risk == 1:
            health_data = {
                "heart_rate": self.rng.uniform(100, 125),
                "blood_oxygen": self.rng.uniform(90, 95),
                "systolic": self.rng.uniform(140, 165),
                "diastolic": self.rng.uniform(90, 105),
                "steps": self.rng.uniform(500, 2000),
            }
            med_data = {"total_medications": self.rng.randint(2, 5),
                        "adherence_rate": self.rng.uniform(0.5, 0.75),
                        "missed_doses": self.rng.randint(1, 3)}
        else:  # risk 2
            health_data = {
                "heart_rate": self.rng.uniform(130, 160),
                "blood_oxygen": self.rng.uniform(80, 92),
                "systolic": self.rng.uniform(165, 200),
                "diastolic": self.rng.uniform(100, 125),
                "steps": self.rng.uniform(0, 500),
            }
            med_data = {"total_medications": self.rng.randint(1, 4),
                        "adherence_rate": self.rng.uniform(0.3, 0.6),
                        "missed_doses": self.rng.randint(2, 5)}

        health_feat = self.health_ext.extract_from_dict(health_data)
        med_feat = self.med_ext.extract_from_dict(med_data)
        return health_feat, med_feat


# ---------------------------------------------------------------------------
# 主预处理流程
# ---------------------------------------------------------------------------

def preprocess(max_per_class: int = 300):
    """
    生成真实特征训练集。
    max_per_class: 每个风险等级最多生成多少样本（控制规模）
    """
    status = check_datasets()
    print("=" * 60)
    print("真实数据预处理")
    print("=" * 60)
    print(f"ESC-50:        {'就绪' if status['esc50'] else '未就绪'} ({status['esc50_wavs']} wav)")
    print(f"Pexels 视频:   {'就绪' if status['pexels_video'] else '未就绪'} "
          f"({status['pexels_video_counts']})")

    if not status["esc50"] and not status["pexels_video"]:
        print("\n[ERROR] 两个数据集均未就绪，无法生成真实特征。")
        print("请先运行: python data/download_datasets.py")
        print("或使用半合成兜底: python data/prepare_data.py (旧版噪声特征)")
        return False

    extractor = UnifiedFeatureExtractor()
    synther = SemiSyntheticHealthMed()

    samples = {"video": [], "audio": [], "health": [], "medication": [], "labels": []}
    count_by_class = {0: 0, 1: 0, 2: 0}

    # ---- 1. Pexels 视频样本（MediaPipe 真实骨架特征）----
    if status["pexels_video"]:
        print("\n[1/3] 处理 Pexels 视频集（MediaPipe PoseLandmarker）...")
        sequences = discover_pexels_videos()
        print(f"  发现 {len(sequences)} 个视频文件")
        print(f"    分布: {status['pexels_video_counts']}")

        esc50_samples = load_esc50_metadata() if status["esc50"] else []
        esc50_by_risk = defaultdict(list)
        for s in esc50_samples:
            esc50_by_risk[s["risk"]].append(s)

        # 缓存：按风险等级缓存已提取的视频特征，供后续 ESC-50/半合成样本配对复用
        video_feat_cache_by_risk = defaultdict(list)
        n_extracted = 0
        n_skipped = 0

        for seq in sequences:
            if count_by_class[seq["risk"]] >= max_per_class:
                continue
            try:
                video_feat = extractor.video.extract_from_file(seq["path"])
                # 跳过全零特征（MediaPipe 未检测到人体的视频）
                if not np.isfinite(video_feat).all() or np.linalg.norm(video_feat) < 1e-6:
                    n_skipped += 1
                    print(f"    [跳过-零特征] {seq['name']}")
                    continue
            except Exception as e:
                n_skipped += 1
                print(f"    [跳过] {seq['name']}: {e}")
                continue

            n_extracted += 1
            risk = seq["risk"]
            video_feat_cache_by_risk[risk].append(video_feat)

            # 配对音频：跌倒配高风险声，日常配低风险声，挣扎配中风险声
            paired_audio_risk = risk
            if esc50_by_risk.get(paired_audio_risk):
                audio_sample = random.choice(esc50_by_risk[paired_audio_risk])
                try:
                    audio_feat = extractor.audio.extract_from_file(audio_sample["path"])
                except Exception:
                    audio_feat = np.zeros(AUDIO_DIM, dtype=np.float32)
            else:
                audio_feat = np.zeros(AUDIO_DIM, dtype=np.float32)

            health_feat, med_feat = synther.generate(risk)

            samples["video"].append(video_feat)
            samples["audio"].append(audio_feat)
            samples["health"].append(health_feat)
            samples["medication"].append(med_feat)
            samples["labels"].append(risk)
            count_by_class[risk] += 1

        print(f"  视频提取: 成功={n_extracted} 跳过={n_skipped}")
        print(f"  视频样本: 低={count_by_class[0]} 中={count_by_class[1]} 高={count_by_class[2]}")
        print(f"  视频特征缓存: " + ", ".join(
            f"risk{r}={len(v)}" for r, v in sorted(video_feat_cache_by_risk.items())))
    else:
        video_feat_cache_by_risk = defaultdict(list)

    # ---- 2. ESC-50 纯音频样本（配对同类真实视频特征）----
    if status["esc50"]:
        print("\n[2/3] 处理 ESC-50 音频样本（配对同类真实视频特征）...")
        esc50_samples = load_esc50_metadata()
        print(f"  可用相关音频: {len(esc50_samples)} 条")

        for s in esc50_samples:
            if count_by_class[s["risk"]] >= max_per_class:
                continue
            try:
                audio_feat = extractor.audio.extract_from_file(s["path"])
            except Exception as e:
                print(f"    [跳过] {s['filename']}: {e}")
                continue

            risk = s["risk"]
            # 配对同类真实视频特征（若有缓存），否则用零向量兜底
            if video_feat_cache_by_risk.get(risk):
                paired_video = random.choice(video_feat_cache_by_risk[risk])
            else:
                # 退而求其次：高风险配 falls，低风险配 daily，中风险配 struggle/daily
                fallback_risk = risk if risk in video_feat_cache_by_risk else (
                    0 if risk == 0 else (2 if risk == 2 else 0))
                if video_feat_cache_by_risk.get(fallback_risk):
                    paired_video = random.choice(video_feat_cache_by_risk[fallback_risk])
                else:
                    paired_video = np.zeros(VIDEO_DIM, dtype=np.float32)

            health_feat, med_feat = synther.generate(risk)

            samples["video"].append(paired_video)
            samples["audio"].append(audio_feat)
            samples["health"].append(health_feat)
            samples["medication"].append(med_feat)
            samples["labels"].append(risk)
            count_by_class[risk] += 1

        print(f"  累计: 低={count_by_class[0]} 中={count_by_class[1]} 高={count_by_class[2]}")

    # ---- 3. 半合成补充样本（各类补满至 max_per_class）----
    needs_synth = any(count_by_class[r] < max_per_class
                      for r in [0, 1, 2]) and (status["esc50"] or status["pexels_video"])
    if needs_synth:
        print("\n[3/3] 补充半合成样本（各类补满）...")
        total_synth = 0
        for risk in [0, 1, 2]:
            needed = max_per_class - count_by_class[risk]
            if needed <= 0:
                continue
            print(f"  风险{risk}: 需补充 {needed} 个")
            # 选用同类真实视频/音频特征作为背景，加轻微扰动
            if video_feat_cache_by_risk.get(risk):
                bg_videos = video_feat_cache_by_risk[risk]
            elif video_feat_cache_by_risk.get(0 if risk != 2 else 2):
                bg_videos = video_feat_cache_by_risk[0 if risk != 2 else 2]
            else:
                bg_videos = [np.zeros(VIDEO_DIM, dtype=np.float32)]

            # 收集已有的同类音频特征作为背景
            bg_audios = []
            for i, lbl in enumerate(samples["labels"]):
                if lbl == risk:
                    bg_audios.append(samples["audio"][i])
            if not bg_audios:
                bg_audios = [np.zeros(AUDIO_DIM, dtype=np.float32)]

            for _ in range(needed):
                # 随机选一个同类真实特征 + 轻微扰动（模拟不同场景）
                v = random.choice(bg_videos) + np.random.randn(VIDEO_DIM).astype(np.float32) * 0.02
                a = random.choice(bg_audios) + np.random.randn(AUDIO_DIM).astype(np.float32) * 0.02
                health_feat, med_feat = synther.generate(risk)
                samples["video"].append(v)
                samples["audio"].append(a)
                samples["health"].append(health_feat)
                samples["medication"].append(med_feat)
                samples["labels"].append(risk)
                count_by_class[risk] += 1
                total_synth += 1

        print(f"  半合成补充 {total_synth} 个")
        print(f"  补充后: 低={count_by_class[0]} 中={count_by_class[1]} 高={count_by_class[2]}")

    # ---- 汇总 & 保存 ----
    n = len(samples["labels"])
    if n == 0:
        print("\n[ERROR] 未生成任何样本")
        return False

    for key in ["video", "audio", "health", "medication"]:
        samples[key] = np.array(samples[key], dtype=np.float32)
    samples["labels"] = np.array(samples["labels"], dtype=np.int64)

    # 打乱
    idx = np.random.permutation(n)
    for key in ["video", "audio", "health", "medication", "labels"]:
        samples[key] = samples[key][idx]

    print(f"\n[汇总] 总样本数: {n}")
    print(f"  标签分布: {np.bincount(samples['labels'])}")
    print(f"  video: {samples['video'].shape}  audio: {samples['audio'].shape}")
    print(f"  health: {samples['health'].shape}  med: {samples['medication'].shape}")

    # 切分
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    splits = {
        "train": idx[:train_end],
        "val": idx[train_end:val_end],
        "test": idx[val_end:],
    }

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for split_name, split_idx in splits.items():
        split_dir = os.path.join(PROCESSED_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for key in ["video", "audio", "health", "medication"]:
            np.save(os.path.join(split_dir, f"{key}_features.npy"), samples[key][split_idx])
        np.save(os.path.join(split_dir, "labels.npy"), samples["labels"][split_idx])
        print(f"  {split_name}: {len(split_idx)} 样本")

    # 保存元数据
    meta = {
        "generated_at": datetime.now().isoformat(),
        "total_samples": n,
        "label_distribution": {str(k): int(v) for k, v in enumerate(np.bincount(samples['labels']))},
        "feature_dims": {"video": VIDEO_DIM, "audio": AUDIO_DIM,
                         "health": HEALTH_DIM, "medication": MED_DIM},
        "data_sources": {
            "esc50_ready": status["esc50"],
            "pexels_video_ready": status["pexels_video"],
            "pexels_video_counts": status["pexels_video_counts"],
            "video_feature_cache": {str(r): len(v) for r, v in sorted(video_feat_cache_by_risk.items())},
            # 兼容旧字段
            "urfd_ready": status["pexels_video"],
        },
        "split": {"train": train_end, "val": val_end - train_end, "test": n - val_end},
    }
    with open(os.path.join(PROCESSED_DIR, "metadata_real.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] 真实特征数据集已保存到 {PROCESSED_DIR}")
    print("下一步: python train.py  (重训)")
    return True


def main():
    parser = argparse.ArgumentParser(description="真实数据预处理")
    parser.add_argument("--max_per_class", type=int, default=300,
                        help="每个风险等级最大样本数")
    parser.add_argument("--check_only", action="store_true",
                        help="仅检查数据集就绪状态")
    args = parser.parse_args()

    if args.check_only:
        status = check_datasets()
        print("数据集就绪状态:")
        print(f"  ESC-50: {'就绪' if status['esc50'] else '未就绪'} ({status['esc50_wavs']} wav)")
        print(f"  URFD:   {'就绪' if status['urfd'] else '未就绪'} ({status['urfd_sequences']} 序列)")
        return

    preprocess(max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
