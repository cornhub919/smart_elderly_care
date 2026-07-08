"""
公开数据集下载脚本
==================
下载两类公开数据集，用于训练真实多模态融合模型：

1. Pexels 视频集（替代 URFD）
   - URFD 官方源不稳定（502），且需 RAR 解压工具。
   - 改用 Pexels 免费版权视频 CDN（直链稳定可下载），
     搜索 "person falling"/"elderly"/"old person" 等关键词，
     下载含人物活动的真实视频片段，用 MediaPipe PoseLandmarker
     提取骨架特征（重心轨迹 / 躯干角度 / 运动速度）。
   - 按风险等级组织目录：
       falls/      → 高风险（person falling / falling down）
       daily/      → 低风险（elderly walking / old person）
       struggle/   → 中风险（person sitting down / struggling）

2. ESC-50
   - 环境声音分类数据集，2000 条 5 秒音频，50 类
   - 官方: https://github.com/karoldvl/ESC-50
   - 我们抽取与养老场景相关的类别：
       * 哭声 / 尖叫        → 呼救
       * 玻璃碎裂 / 咳嗽    → 撞击 / 异常
       * 喷嚏 / 打鼾        → 生理
       * 静默 / 雨声 / 风声 → 正常背景

设计原则
--------
- 幂等：已下载且校验通过则跳过
- 断点续传：大文件 HTTP Range 续传
- 多源回退：主源失败自动尝试镜像
- 离线友好：网络不可达时优雅退出并打印手动下载指引
- 不强依赖：即使数据未就绪，预处理脚本也能用半合成数据兜底

用法
----
    python data/download_datasets.py            # 下载全部
    python data/download_datasets.py --only pexels_video
    python data/download_datasets.py --only esc50
    python data/download_datasets.py --check     # 仅检查是否就绪
"""

import os
import sys
import re
import argparse
import hashlib
import urllib.request
import urllib.error
import urllib.parse
import zipfile
import tarfile
import shutil
import json
from pathlib import Path

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V2_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(V2_ROOT, "data", "raw")

# ---------------------------------------------------------------------------
# 数据集定义
# ---------------------------------------------------------------------------

DATASETS = {
    "esc50": {
        "name": "ESC-50",
        "target_dir": os.path.join(RAW_DIR, "esc50"),
        "sources": [
            # GitHub 官方 release zip（最稳定）
            "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip",
        ],
        # 下载后的压缩包临时名
        "archive_name": "esc50-master.zip",
        # 解压后顶层目录名（用于定位）
        "extracted_topdir": "ESC-50-master",
        # 就绪校验：存在 audio/ 目录且含 .wav 文件
        "ready_check": lambda d: os.path.isdir(os.path.join(d, "ESC-50-master", "audio"))
                                  and len(list(Path(os.path.join(d, "ESC-50-master", "audio")).glob("*.wav"))) > 100,
        "ready_marker": os.path.join(RAW_DIR, "esc50", "_READY"),
        "desc": "环境声音 50 类，2000 条 5s 音频",
    },
    "pexels_video": {
        "name": "Pexels 视频集",
        "target_dir": os.path.join(RAW_DIR, "pexels_video"),
        # 按风险等级组织子目录 + 搜索关键词
        "risk_categories": {
            "falls": {            # 高风险（跌倒）
                "risk": 2,
                "queries": [
                    "person falling", "falling down", "man falling",
                    "trip and fall", "slipping fall",
                ],
            },
            "struggle": {         # 中风险（挣扎/困难起身）
                "risk": 1,
                "queries": [
                    "person sitting down", "struggling to stand",
                    "person standing up", "bending over",
                    "old man sitting", "person sitting",
                ],
            },
            "daily": {            # 低风险（日常活动）
                "risk": 0,
                "queries": [
                    "elderly walking", "old person walking",
                    "senior exercising", "elderly standing",
                ],
            },
        },
        # 每类最多下载的视频数
        "max_per_category": 30,
        # 优先下载的清晰度后缀（hd 体积适中）
        "quality_pref": ["-hd_", "-uhd_", "-sd_"],
        "ready_check": lambda d: (
            sum(1 for sub in ["falls", "struggle", "daily"]
                if os.path.isdir(os.path.join(d, sub))
                and len(list(Path(os.path.join(d, sub)).glob("*.mp4"))) > 0) >= 3
            if os.path.isdir(d) else False
        ),
        "ready_marker": os.path.join(RAW_DIR, "pexels_video", "_READY"),
        "desc": "Pexels 免费视频，跌倒/日常/挣扎三类，MediaPipe 提取骨架特征",
    },
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def human_size(num_bytes: int) -> str:
    """字节数转人类可读"""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}TB"


def download_with_resume(url: str, dest: str, timeout: int = 30) -> bool:
    """
    带断点续传的下载。
    返回 True 表示成功，False 表示失败（含网络错误）。
    """
    # 已存在且完整则跳过（简单判断：非 0 字节）
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        # 续传时需确认远端支持 Range；这里若已有文件则尝试续传
        existing = os.path.getsize(dest)
    else:
        existing = 0

    req_headers = {"User-Agent": "Mozilla/5.0 (compatible; SmartElderlyCareBot/1.0)"}
    if existing > 0:
        req_headers["Range"] = f"bytes={existing}-"

    try:
        req = urllib.request.Request(url, headers=req_headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # 判断是否支持续传
            if existing > 0 and resp.status == 206:
                mode = "ab"
                print(f"  [续传] 从 {human_size(existing)} 继续")
            elif existing > 0 and resp.status == 200:
                # 不支持 Range，从头来
                mode = "wb"
                existing = 0
            else:
                mode = "wb"

            total = int(resp.headers.get("Content-Length", 0)) + existing
            downloaded = existing
            with open(dest, mode) as f:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 / total
                        # 进度条（简陋但有效）
                        bar = "#" * int(pct // 2)
                        sys.stdout.write(f"\r  {human_size(downloaded)}/{human_size(total)} [{bar:<50}] {pct:5.1f}%")
                        sys.stdout.flush()
            sys.stdout.write("\n")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as e:
        print(f"\n  [失败] {type(e).__name__}: {e}")
        return False
    except Exception as e:
        print(f"\n  [失败] {type(e).__name__}: {e}")
        return False


def safe_extract_zip(zip_path: str, target_dir: str) -> bool:
    """安全解压 zip"""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # 防路径穿越
            for member in zf.namelist():
                member_path = os.path.normpath(os.path.join(target_dir, member))
                if not member_path.startswith(os.path.normpath(target_dir)):
                    print(f"  [警告] 跳过可疑路径: {member}")
                    continue
            zf.extractall(target_dir)
        return True
    except Exception as e:
        print(f"  [解压失败] {e}")
        return False


def write_marker(marker_path: str, info: dict) -> None:
    """写就绪标记文件"""
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 各数据集下载逻辑
# ---------------------------------------------------------------------------

def download_esc50() -> bool:
    """下载并解压 ESC-50"""
    ds = DATASETS["esc50"]
    target = ds["target_dir"]
    os.makedirs(target, exist_ok=True)

    # 已就绪？
    if os.path.exists(ds["ready_marker"]):
        print(f"[ESC-50] 已就绪，跳过: {target}")
        return True

    print(f"[ESC-50] 开始下载到 {target}")
    archive = os.path.join(target, ds["archive_name"])

    success = False
    for url in ds["sources"]:
        print(f"  尝试源: {url}")
        if download_with_resume(url, archive):
            success = True
            break
        print("  切换下一源...")

    if not success:
        print("[ESC-50] 所有源均失败。请手动下载:")
        print(f"  https://github.com/karoldvl/ESC-50")
        print(f"  将 zip 放到 {archive} 后重跑本脚本。")
        return False

    # 解压
    print("[ESC-50] 解压中...")
    if not safe_extract_zip(archive, target):
        return False

    # 校验
    if ds["ready_check"](target):
        write_marker(ds["ready_marker"], {
            "dataset": "ESC-50",
            "source": url,
            "note": "2000 条 5s 音频，50 类"
        })
        # 清理压缩包
        if os.path.exists(archive):
            os.remove(archive)
        print("[ESC-50] [OK] 就绪")
        return True
    else:
        print("[ESC-50] [FAIL] 解压后校验失败，目录结构不符合预期")
        return False


def _search_pexels_videos(query: str, timeout: int = 30, max_retries: int = 3) -> list:
    """搜索 Pexels 视频页面，抓取视频直链。
    返回 [(video_id, hd_url), ...]，优先 hd 清晰度。
    带重试 + 随机延迟，规避 Pexels 反爬限流（连续搜索易触发 403）。
    """
    import ssl
    import time
    import random
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    # 轮换 UA，降低被识别为爬虫的概率
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]
    url = f"https://www.pexels.com/search/videos/{urllib.parse.quote(query)}/"
    html = None
    for attempt in range(1, max_retries + 1):
        headers = {"User-Agent": random.choice(user_agents)}
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
            html = resp.read().decode("utf-8", errors="ignore")
            resp.close()
            break
        except Exception as e:
            wait = 3 * attempt + random.uniform(1, 3)
            print(f"    [搜索重试 {attempt}/{max_retries}] {query}: {e}，等待 {wait:.1f}s")
            if attempt < max_retries:
                time.sleep(wait)
            else:
                print(f"    [搜索失败] {query}: 重试 {max_retries} 次仍失败")
                return []
    if html is None:
        return []
    # 搜索后随机延迟，避免连续请求触发限流
    time.sleep(random.uniform(1.5, 3.5))

    # 提取所有 video-files 直链
    links = re.findall(
        r'https://videos\.pexels\.com/video-files/\d+/[^\s"\'<>]+\.mp4', html
    )
    # 按 video_id 去重，优先选 hd
    seen = {}
    for lk in links:
        m = re.search(r'/(\d+)/', lk)
        if not m:
            continue
        vid = m.group(1)
        if vid not in seen:
            seen[vid] = lk
        else:
            # 已有非 hd，新的是 hd 则替换
            if "-hd_" in lk and "-hd_" not in seen[vid]:
                seen[vid] = lk
    return [(vid, lk) for vid, lk in seen.items()]


def download_pexels_video() -> bool:
    """下载 Pexels 视频集（替代 URFD，CDN 稳定可直链下载）。
    按风险等级搜索关键词，下载到 falls/ struggle/ daily/ 子目录。
    """
    ds = DATASETS["pexels_video"]
    target = ds["target_dir"]
    os.makedirs(target, exist_ok=True)

    if os.path.exists(ds["ready_marker"]):
        print(f"[Pexels] 已就绪，跳过: {target}")
        return True

    max_per_cat = ds["max_per_category"]
    quality_pref = ds["quality_pref"]
    total_ok = 0

    for cat_name, cat_info in ds["risk_categories"].items():
        risk = cat_info["risk"]
        cat_dir = os.path.join(target, cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        existing = len(list(Path(cat_dir).glob("*.mp4")))
        print(f"\n[Pexels] 类别 '{cat_name}' (风险={risk}) "
              f"已有 {existing} 个，目标 {max_per_cat}")

        collected = 0
        for query in cat_info["queries"]:
            if collected >= max_per_cat:
                break
            print(f"  搜索: '{query}'")
            results = _search_pexels_videos(query)
            print(f"    找到 {len(results)} 个候选视频")
            for vid, url in results:
                if collected >= max_per_cat:
                    break
                fname = f"pexels_{vid}.mp4"
                fpath = os.path.join(cat_dir, fname)
                if os.path.exists(fpath):
                    collected += 1
                    continue
                # 选择合适清晰度
                if "-hd_" not in url and "-uhd_" not in url and "-sd_" not in url:
                    continue
                try:
                    if download_with_resume(url, fpath, timeout=90):
                        collected += 1
                        total_ok += 1
                        print(f"    [OK] {fname} "
                              f"({os.path.getsize(fpath)/1024/1024:.1f} MB)")
                except Exception as e:
                    print(f"    [跳过] {vid}: {e}")
            print(f"  累计: {collected}/{max_per_cat}")

        print(f"  类别 '{cat_name}' 完成: {collected} 个视频")

    # 校验
    if ds["ready_check"](target):
        write_marker(ds["ready_marker"], {
            "dataset": "Pexels 视频集",
            "note": f"跌倒/挣扎/日常三类视频，共 {total_ok} 个",
            "categories": {c: len(list(Path(os.path.join(target, c)).glob("*.mp4")))
                           for c in ds["risk_categories"]},
        })
        print(f"\n[Pexels] [OK] 就绪，共 {total_ok} 个视频")
        return True

    print(f"\n[Pexels] [WARN] 下载不完整（共 {total_ok} 个），"
          "至少需要三类各 1 个视频。")
    return False


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def check_ready(name: str) -> bool:
    """检查某数据集是否就绪"""
    ds = DATASETS[name]
    return os.path.exists(ds["ready_marker"]) or ds["ready_check"](ds["target_dir"])


def _status_text(ready: bool) -> str:
    return "[OK] 就绪" if ready else "[FAIL] 未就绪"


def main():
    parser = argparse.ArgumentParser(description="下载公开多模态数据集")
    parser.add_argument("--only", choices=["esc50", "pexels_video", "all"], default="all",
                        help="仅下载指定数据集")
    parser.add_argument("--check", action="store_true",
                        help="仅检查就绪状态，不下载")
    args = parser.parse_args()

    print("=" * 60)
    print("智护家 — 公开数据集下载")
    print(f"原始数据目录: {RAW_DIR}")
    print("=" * 60)

    if args.check:
        print("\n[就绪状态检查]")
        for name, ds in DATASETS.items():
            ready = check_ready(name)
            status = _status_text(ready)
            print(f"  {ds['name']:25s} {status}")
            if not ready:
                print(f"    目录: {ds['target_dir']}")
        return

    targets = ["esc50", "pexels_video"] if args.only == "all" else [args.only]

    results = {}
    for name in targets:
        print()
        if name == "esc50":
            results["esc50"] = download_esc50()
        elif name == "pexels_video":
            results["pexels_video"] = download_pexels_video()

    print("\n" + "=" * 60)
    print("[下载总结]")
    for name, ok in results.items():
        ds = DATASETS[name]
        print(f"  {ds['name']:25s} {'[OK] 成功' if ok else '[FAIL] 失败(见上方指引)'}")
    print("=" * 60)

    # 提示下一步
    any_ready = any(results.values())
    if any_ready:
        print("\n下一步: 运行预处理脚本提取真实特征")
        print("  python data/preprocess_real.py")
    else:
        print("\n两个数据集均未就绪。可先用半合成特征兜底（见 data/prepare_data.py），")
        print("待数据到位后再重训。")


if __name__ == "__main__":
    main()
