# 在 Google Colab 中运行的下载脚本：补齐「当前版本全集」的卫星星历部分。
#
# 方案 A = 方案 B（已下载）+ 本脚本的 spk/satellites 当前版（107 文件，约 29.7GB）。
# 29.7GB 超过免费 Drive 15GB，故按 10GB 贪心分箱；每跑一箱 → 搬到本地 → 删 Drive
# 上的该箱数据 → 改 BATCH 数字跑下一箱，直到全部搬完。
#
# 自包含：不依赖 Drive 上任何旧清单文件——本脚本自行枚举 satellites 目录、
# HEAD 取真实大小、分箱。不会重复下载方案 B 的文件（那些不在 satellites 目录）。
#
# 用法：把 BATCH 改成当前要跑的箱号（1 起），整个文件复制到 Colab 单元格运行。
#       产物：<Drive>/naif_download/generic_kernels/spk/satellites/*.bsp

import os
import re
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from google.colab import drive

# ====== 参数：每跑一箱改这里 ======
BATCH = 1  # 箱号，从 1 开始
BATCH_LIMIT_MB = 10000  # 每箱上限 10GB（Drive 免费额度 15GB 留余量）
# ==================================

drive.mount("/content/drive")

DIR_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/"
OUT_DIR = "/content/drive/MyDrive/naif_download/generic_kernels/spk/satellites"
SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)([KMG])?$")
_UNIT = {"K": 1024, "M": 1024**2, "G": 1024**3}

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (kernel downloader)"


def fetch_sizes():
    """枚举 satellites 目录文件（跳过 a_old_versions/）并 HEAD 取真实大小。"""
    r = session.get(DIR_URL, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    names = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        name = a.get_text(strip=True)
        if href.startswith(("/", "?", "http", "mailto:")) or href in ("../",):
            continue
        if href.endswith("/") or not name:
            continue
        names.append(name)
    print(f"satellites 目录：{len(names)} 个文件，HEAD 取大小中……")
    entries = []
    for i, name in enumerate(sorted(names)):
        url = DIR_URL + quote(name)
        size = -1
        try:
            h = session.head(url, timeout=30, allow_redirects=True)
            if h.status_code == 405:
                g = session.get(url, stream=True, timeout=30)
                cl = g.headers.get("Content-Length")
                g.close()
                size = int(cl) if cl and cl.isdigit() else -1
            else:
                h.raise_for_status()
                cl = h.headers.get("Content-Length")
                size = int(cl) if cl and cl.isdigit() else -1
        except Exception:
            size = -1
        entries.append((name, size))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(names)}")
        time.sleep(0.05)
    return entries


def make_bins(entries, limit_mb):
    """按大小降序贪心装箱，每箱 ≤ limit_mb。"""
    limit = limit_mb * 1048576
    entries = sorted(entries, key=lambda x: -(x[1] if x[1] > 0 else 0))
    bins, cur, cur_sz = [], [], 0
    for e in entries:
        s = e[1] if e[1] > 0 else 0
        if cur_sz + s > limit and cur:
            bins.append(cur)
            cur, cur_sz = [], 0
        cur.append(e)
        cur_sz += s
    if cur:
        bins.append(cur)
    return bins


def download(url, dest, expect_size):
    tmp = dest + ".part"
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        size = os.path.getsize(tmp)
        if expect_size > 0 and size != expect_size:
            os.remove(tmp)
            return False, f"大小不符（得 {size}，期望 {expect_size}）"
        os.replace(tmp, dest)
        return True, ""
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return False, str(e)


entries = fetch_sizes()
bins = make_bins(entries, BATCH_LIMIT_MB)
total = sum(s for _, s in entries if s > 0)
print(
    f"\n全部 {len(entries)} 文件，{total / 1073741824:.2f} GB，"
    f"分 {len(bins)} 箱（每箱 ≤{BATCH_LIMIT_MB / 1000:.0f}GB）："
)
for i, b in enumerate(bins):
    sz = sum(s for _, s in b if s > 0) / 1073741824
    print(f"  箱 {i + 1}: {len(b)} 文件  {sz:.2f} GB" + ("  ← 本批" if i + 1 == BATCH else ""))

if not (1 <= BATCH <= len(bins)):
    raise SystemExit(f"BATCH={BATCH} 超出范围 1..{len(bins)}")

batch = bins[BATCH - 1]
batch_sz = sum(s for _, s in batch if s > 0)
print(f"\n开始下载箱 {BATCH}（{batch_sz / 1073741824:.2f} GB）→ {OUT_DIR}")

os.makedirs(OUT_DIR, exist_ok=True)
ok = skipped = 0
failed = []
done_bytes = 0
start = time.time()
for i, (name, size) in enumerate(batch):
    dest = os.path.join(OUT_DIR, name)
    if os.path.exists(dest):
        got = os.path.getsize(dest)
        if size <= 0 or got == size:
            skipped += 1
            continue
        os.remove(dest)
    o, err = download(DIR_URL + quote(name), dest, size)
    if o:
        ok += 1
        done_bytes += os.path.getsize(dest)
    else:
        failed.append((name, err))
    # 每完成一个文件打印一次（箱内文件数少，实时反馈更重要）
    el = time.time() - start
    speed = done_bytes / el / 1048576 if el > 0 else 0
    print(
        f"  {i + 1}/{len(batch)} {name}  "
        f"已下 {done_bytes / 1073741824:.2f} GB  速度 {speed:.1f} MB/s  失败 {len(failed)}"
    )

print(f"\n箱 {BATCH} 完成：下载 {ok}，跳过 {skipped}，失败 {len(failed)}")
for n, e in failed:
    print(f"  ❌ {n}: {e}")
print("下一步：把本箱搬到本地 → 删 Drive 上 naif_download/ 的本箱数据 → BATCH 加 1 重跑")
