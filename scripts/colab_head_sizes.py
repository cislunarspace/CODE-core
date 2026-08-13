# 在 Google Colab 中运行：对已枚举的 NAIF 文件清单逐个发请求取真实大小。
#
# 背景：NAIF 目录索引页的 HTML 非标准结构，文件大小解析两轮都失败；改用
# 直接发 HTTP 请求拿 Content-Length（HEAD 优先，405 时回退 GET stream 只取头
# 不读 body）。输入是 colab_list_naif.py 产出的 Drive 文件
# naif_inventory/naif_files.csv（路径,文件名,字节），输出同目录
# naif_files_sized.csv（补真实字节数）。
#
# 1145 个文件 × ~0.1s 延迟 + 网络往返，约 10-20 分钟，进度每 50 个打印一次。
# 失败的文件 size 记 -1（下载阶段再处理）。
#
# 用法：全部复制到 Colab 单元格运行。

import csv
import os
import time

import requests
from google.colab import drive

drive.mount("/content/drive")

IN_DIR = "/content/drive/MyDrive/naif_inventory"
IN_CSV = os.path.join(IN_DIR, "naif_files.csv")
OUT_CSV = os.path.join(IN_DIR, "naif_files_sized.csv")
ROOT_URL = "https://naif.jpl.nasa.gov/pub/naif/"

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (inventory script)"


def fetch_size(url):
    """取文件大小：HEAD 优先，405/501 回退 GET stream 只读头。失败返回 -1。"""
    try:
        r = session.head(url, timeout=30, allow_redirects=True)
        if r.status_code == 405:
            r = session.get(url, stream=True, timeout=30)
            cl = r.headers.get("Content-Length")
            r.close()
            return int(cl) if cl and cl.isdigit() else -1
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        return int(cl) if cl and cl.isdigit() else -1
    except Exception:
        return -1


rows = []
with open(IN_CSV) as f:
    rows = list(csv.reader(f))
header, body = rows[0], rows[1:]
print(f"共 {len(body)} 个文件，开始取大小……")

ok, failed = 0, 0
start = time.time()
for i, row in enumerate(body):
    path, name, _ = row
    # 修双斜杠：枚举阶段目录名已含尾斜杠、拼接多了一个
    clean = path.replace("//", "/")
    size = fetch_size(ROOT_URL + clean)
    row[2] = str(size)
    if size < 0:
        failed += 1
    else:
        ok += 1
    if (i + 1) % 50 == 0:
        el = time.time() - start
        print(f"  {i + 1}/{len(body)}（成功 {ok}，失败 {failed}，用时 {el:.0f}s）")
    time.sleep(0.05)

with open(OUT_CSV, "w", newline="") as f:
    csv.writer(f).writerows([header] + body)

total = sum(int(r[2]) for r in body if r[2].isdigit() and int(r[2]) > 0)
print(f"\n完成：成功 {ok}，失败 {failed}，总大小 {total / 1073741824:.2f} GB")
print(f"输出: {OUT_CSV}")
