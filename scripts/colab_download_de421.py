# 在 Google Colab 中运行的下载脚本：从 NAIF 官网下载 DE421 行星星历内核。
#
# 用途：本机（国内网络）直连 naif.jpl.nasa.gov 不可达，但 Colab 可访问。本脚本
# 把 de421.bsp 与 de421.cmt 存到 Google Drive，再从 Drive 取回本地。
#
# 背景（issue #665）：CE-5 精密定轨策略（孔静等 2022，表 1）用 JPL DE421 作第三
# 体星历。仓库 kernels/ 只有 de440s/de430，kernels-v1 release 也没有 de421，
# download_kernels.py（无差别拉取 release 全部资产）因此拿不到它，需从 NAIF 源补。
#
# 下载内容（DE421 位于 NAIF 的 a_old_versions/ 历史目录）：
#   1. de421.bsp  JPL 行星星历内核本体，16,790,528 字节（大小来源：naif_files_sized.csv）
#      主: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de421.bsp
#      备: https://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de421.bsp
#   2. de421.cmt  内核注释（1,385 字节）：含段清单、时间覆盖与精度说明
#      主: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de421.cmt
#
# 注意：DE421 的月球姿态/天平动内核 moon_pa_de421_1900-2050.bpc（即项目里的
# SPICELunaCurrentKernel.bpc）与 DE421 IOM 报告均已另行取得，本脚本不重复下载。
#
# 用法：全部复制到 Colab 单元格运行。产物：<Drive>/e2m2e_de421/ 下两个文件。
# 可选：末尾 CELL 2 用 spiceypy 验核（确认能加载、打印时间覆盖），需先装 spiceypy。

import os

import requests
from google.colab import drive

# 1. 挂载 Google Drive
drive.mount("/content/drive")


def download_file(filename, save_dir, urls, expect_size=None, magic=None):
    """按顺序尝试 urls，下载到临时文件再改名落盘。

    中断的下载不残留半截文件；已存在且校验通过则跳过（幂等，可反复重跑）。
    expect_size / magic 任一不符即删除并报错，绝不把坏内核留在 Drive 上。
    """
    save_path = os.path.join(save_dir, filename)
    tmp_path = save_path + ".part"

    def _verify(path):
        """返回 (ok, 说明)；magic 为 (偏移, 期望字节) 二元组。"""
        if expect_size is not None and os.path.getsize(path) != expect_size:
            return False, f"大小不符（得 {os.path.getsize(path)}，期望 {expect_size}）"
        if magic is not None:
            offset, expected = magic
            with open(path, "rb") as fh:
                fh.seek(offset)
                got = fh.read(len(expected))
            if got != expected:
                return False, f"文件头不符（得 {got!r}，期望 {expected!r}）"
        return True, ""

    if os.path.exists(save_path):
        ok, why = _verify(save_path)
        if ok:
            size_mb = os.path.getsize(save_path) / 1024 / 1024
            print(f"文件已存在且校验通过，跳过: {filename} ({size_mb:.2f} MB)")
            return True
        print(f"⚠️ 已存在文件校验失败（{why}），重新下载: {filename}")
        os.remove(save_path)

    for url in urls:
        print(f"🚀 正在下载: {filename} ...")
        print(f"   来源: {url}")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            ok, why = _verify(tmp_path)
            if not ok:
                os.remove(tmp_path)
                print(f"⚠️ 该来源文件不可信（{why}），尝试下一个")
                continue
            os.replace(tmp_path, save_path)
            size_mb = os.path.getsize(save_path) / 1024 / 1024
            print(f"✅ 下载完成: {filename} ({size_mb:.2f} MB)")
            return True
        except Exception as e:
            print(f"⚠️ 该来源失败，尝试下一个: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    print(f"❌ 所有来源均下载失败: {filename}")
    return False


# 2. Google Drive 存储目录
out_dir = "/content/drive/MyDrive/e2m2e_de421"
os.makedirs(out_dir, exist_ok=True)

# 3. DE421 内核清单：(保存文件名, 来源列表, 期望字节数, (偏移, 魔数))
#    .bsp 的 DAF/SPK 魔数用 ASCII 表示（不用 bytes.fromhex，规避 Colab 版本差异）。
files = [
    (
        "de421.bsp",
        [
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de421.bsp",
            "https://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de421.bsp",
        ],
        16790528,
        (0, b"DAF/SPK "),
    ),
    (
        "de421.cmt",
        [
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de421.cmt",
        ],
        1385,
        None,
    ),
]

failed = []
for filename, urls, expect_size, magic in files:
    if not download_file(filename, out_dir, urls, expect_size=expect_size, magic=magic):
        failed.append(filename)

print(f"\n🎉 产物目录: {out_dir}/")
for filename, _, _, _ in files:
    path = os.path.join(out_dir, filename)
    if os.path.exists(path):
        print(f"   - {filename}  {os.path.getsize(path)} 字节")
    else:
        print(f"   - {filename}  ❌ 缺失")
if failed:
    print(f"\n⚠️ 失败 {len(failed)} 个：{', '.join(failed)}（重跑本脚本即可，已下好的会跳过）")
print("\n取回本地后放入仓库 kernels/ 目录即可参与 SPICE 加载。")

# ============================================================================
# CELL 2（可选）：用 spiceypy 验核 —— 确认内核真的能加载，并打印时间覆盖。
# 单独一个单元格运行，先装 spiceypy。
# ============================================================================
# !pip -q install spiceypy
#
# import spiceypy as sp
#
# BSP = "/content/drive/MyDrive/e2m2e_de421/de421.bsp"
# sp.kclear()
# sp.furnsh(BSP)
# print("已加载:", sp.ktotal("SPK"), "个 SPK")
#
# # DE421 覆盖 1899-07-29 ~ 2053-10-09；对地月链路关心的现代弧段应完全落在其中。
# ids = sp.spkobj(BSP)
# for body_id in ids:
#     cover = sp.spkcov(BSP, body_id)
#     # 合并重叠窗口后打印，避免极长区间列表
#     win = " — ".join(
#         f"{sp.et2utc(cover[i], 'ISOC', 0)} ~ {sp.et2utc(cover[i + 1], 'ISOC', 0)}"
#         for i in range(0, len(cover), 2)
#     )
#     print(f"  体 {body_id}: {win}")
# sp.kclear()
