# 在 Google Colab 中运行的枚举脚本：爬取 NAIF（JPL）官网目录结构，摸清可下载数据全貌。
#
# 背景：NAIF 全站（pub/naif/）含所有深空任务数据，总量数百 GB；通用内核
# generic_kernels/ 才是日常可用的全集（闰秒/常数/星历/帧/数字形状/恒星等，
# 仪器与姿态内核按任务放在各任务目录下，不在 generic_kernels）。
# 本脚本先枚举、不下载：递归解析 Apache 索引页，输出
#   - naif_tree.txt：目录树（每目录文件数/总大小）
#   - naif_files.csv：文件级清单（相对路径,文件名,字节）
# 存到 Google Drive，据此确定下载范围。
#
# 用法：全部复制到 Colab 单元格运行。

import csv
import os
import re
import time

import requests
from bs4 import BeautifulSoup
from google.colab import drive

drive.mount("/content/drive")

ROOT_URL = "https://naif.jpl.nasa.gov/pub/naif/"
OUT_DIR = "/content/drive/MyDrive/naif_inventory"
os.makedirs(OUT_DIR, exist_ok=True)

# Apache 索引 Size 列格式：纯字节数，或数字 + K/M/G 后缀；"-" 是目录
SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)([KMG])?$")
_UNIT = {"K": 1024, "M": 1024**2, "G": 1024**3}

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (inventory script)"


def parse_size(td):
    """从 td 文本解析字节数；不是大小格式（日期等）返回 None。"""
    txt = td.get_text(strip=True)
    m = SIZE_RE.match(txt)
    if not m:
        return None
    num = float(m.group(1))
    return int(num * _UNIT.get(m.group(2), 1))


def extract_size(a):
    """从 <a> 节点提取文件大小，兼容两种 Apache 索引格式：
    - 表格（<tr><td>…大小列）：遍历 td 找 SIZE_RE 匹配项；
    - 纯文本列表（<pre> 行）：收集 <a> 之后到行尾的文本，取最后一个 K/M/G token。
    取不到返回 0。"""
    tr = a.find_parent("tr")
    if tr is not None:
        for td in tr.find_all("td"):
            s = parse_size(td)
            if s is not None:
                return s
        return 0
    tail = []
    node = a
    while True:
        nxt = node.next_sibling
        if nxt is None:
            break
        text = str(nxt)
        tail.append(text)
        if "\n" in text:
            break
        node = nxt
    m = re.findall(r"(\d+(?:\.\d+)?)([KMG])(?![\d.])", "".join(tail))
    if m:
        num, unit = m[-1]
        return int(float(num) * _UNIT[unit])
    return 0


def fetch_listing(url):
    """抓取 Apache 索引页，返回 (子目录名列表, 文件[(名, 字节)]列表)。"""
    r = session.get(url, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    dirs, files = [], []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        name = a.get_text(strip=True)
        # 只收纯相对路径：跳过站点根/绝对 URL/mailto（坏链接或站外链接）
        if href.startswith(("/", "?", "http", "mailto:")) or href in ("../",):
            continue
        if href.endswith("/") and name:
            dirs.append(name)
        elif name:
            files.append((name, extract_size(a)))
    return sorted(dirs), sorted(files)


lines = []
csv_rows = []
total_files = 0
total_bytes = 0
req_count = 0


def walk(url, depth, max_depth, prefix=""):
    """递归枚举目录树。max_depth 限制深度；404/403 容错不中断。"""
    global total_files, total_bytes, req_count
    req_count += 1
    try:
        dirs, files = fetch_listing(url)
    except requests.HTTPError as e:
        lines.append(f"{prefix}!! HTTP {e.response.status_code}: {url}（跳过）")
        print(f"跳过 {url}: HTTP {e.response.status_code}")
        return
    dir_bytes = sum(s for _, s in files)
    dir_count = len(files)
    total_files += dir_count
    total_bytes += dir_bytes
    rel = url[len(ROOT_URL) :].rstrip("/")
    for fname, fsize in files:
        csv_rows.append((rel + "/" + fname, fname, fsize))
    lines.append(
        f"{prefix}{os.path.basename(url.rstrip('/')) or '/'}/  "
        f"({dir_count} 文件, {dir_bytes / 1048576:.1f} MB, 子目录 {len(dirs)})"
    )
    if depth >= max_depth:
        return
    for d in dirs:
        walk(url + d + "/", depth + 1, max_depth, prefix + "  ")
    time.sleep(0.1)


# 1) 顶层：只列目录名与文件概况，不深入（任务数据太大）
lines.append("=== pub/naif/ 顶层 ===")
top_dirs, top_files = fetch_listing(ROOT_URL)
for d in top_dirs:
    lines.append(f"  {d}/")
for name, size in top_files[:50]:
    lines.append(f"  {name} ({size / 1048576:.1f} MB)")

# 2) generic_kernels/：完整递归（含文件数/大小统计）
print("爬取 generic_kernels/ 目录树中……")
lines.append("")
lines.append("=== generic_kernels/ 完整树（目录: 文件数, 大小, 子目录数）===")
walk(ROOT_URL + "generic_kernels/", 0, 4)

lines.append("")
lines.append(
    f"=== 统计：generic_kernels 共 {total_files} 文件，"
    f"{total_bytes / 1048576 / 1024:.1f} GB，请求 {req_count} 次 ==="
)

out_tree = os.path.join(OUT_DIR, "naif_tree.txt")
with open(out_tree, "w") as f:
    f.write("\n".join(lines) + "\n")
out_csv = os.path.join(OUT_DIR, "naif_files.csv")
with open(out_csv, "w", newline="") as f:
    csv.writer(f).writerows([("path", "name", "bytes")] + csv_rows)
print(f"清单已保存: {out_tree}")
print(f"文件级清单: {out_csv}")

# 控制台打印概况（完整树在文件里）
print("\n".join(lines[:40]))
print("……（完整清单见 Drive 文件）")
