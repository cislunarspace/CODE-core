# 在 Google Colab 中运行的下载脚本：按方案 B 口径下载 NAIF 通用内核全集。
#
# 方案 B：当前版本全集，除卫星星历（spk/satellites 是大头 29.7GB，多为外行星卫星）
#         与历史版本（a_old_versions）。预期 220 文件、9.80 GB，塞进 15GB Drive。
#
# 输入：Drive 上 naif_inventory/naif_files_sized.csv（colab_list_naif.py +
#       colab_head_sizes.py 的产物，含路径与真实字节数）。
# 输出：<Drive>/naif_download/generic_kernels/...（按 NAIF 相对路径建目录）。
#
# 幂等：已存在且字节数匹配的文件跳过；中断后重跑续传。速度与 ETA 每 10 个文件打印。
# 用法：全部复制到 Colab 单元格运行。

import csv
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests
from google.colab import drive

drive.mount("/content/drive")

IN_DIR = "/content/drive/MyDrive/naif_inventory"
IN_CSV = os.path.join(IN_DIR, "naif_files_sized.csv")
OUT_DIR = "/content/drive/MyDrive/naif_download"
ROOT_URL = "https://naif.jpl.nasa.gov/pub/naif/"

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (kernel downloader)"


def include(path):
    """方案 B 过滤：跳过历史版本与卫星星历。"""
    parts = [x for x in path.split("/") if x]
    return not (
        "a_old_versions" in parts
        or (
            len(parts) >= 3
            and parts[0] == "generic_kernels"
            and parts[1] == "spk"
            and parts[2] == "satellites"
        )
    )


def download(url, dest, expect_size):
    """下载到 .part 再改名；返回 (ok, 错误信息)。"""
    tmp = dest + ".part"
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with Path(tmp).open("wb") as f:
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


rows = []
with open(IN_CSV) as f:
    rows = list(csv.reader(f))
body = [r for r in rows[1:] if include(r[0])]
expect_total = sum(int(r[2]) for r in body if r[2].isdigit() and int(r[2]) > 0)
print(f"方案 B：{len(body)} 个文件，预期 {expect_total / 1073741824:.2f} GB")
print(f"输出目录: {OUT_DIR}\n")

done_bytes = 0
skipped = 0
ok = 0
failed = []
start = time.time()
for i, row in enumerate(body):
    path, name, size_s = row
    size = int(size_s) if size_s.isdigit() else -1
    clean = "/".join(x for x in path.split("/") if x)
    dest = os.path.join(OUT_DIR, *clean.split("/"))
    # 幂等：字节数匹配则跳过
    if os.path.exists(dest):
        got = os.path.getsize(dest)
        if size <= 0 or got == size:
            skipped += 1
            continue
        os.remove(dest)  # 大小不符，重下
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = ROOT_URL + "/".join(quote(x, safe="._-") for x in clean.split("/"))
    o, err = download(url, dest, size)
    if o:
        ok += 1
        done_bytes += os.path.getsize(dest)
    else:
        failed.append((path, err))
    if (i + 1) % 10 == 0:
        el = time.time() - start
        speed = done_bytes / el / 1048576 if el > 0 else 0
        eta = (expect_total - done_bytes) / (speed * 1048576) if speed > 0 else -1
        print(
            f"  {i + 1}/{len(body)}  已下 {done_bytes / 1073741824:.2f} GB  "
            f"速度 {speed:.1f} MB/s  ETA {eta / 60:.0f} 分钟  失败 {len(failed)}"
        )
    time.sleep(0.05)

print(f"\n完成：下载 {ok}，跳过 {skipped}，失败 {len(failed)}")
for p, e in failed:
    print(f"  ❌ {p}: {e}")
print("（失败的文件重跑本脚本即可续传）")
