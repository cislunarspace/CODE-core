# 在 Google Colab 中运行的下载脚本：从 NAIF（JPL）官网下载 SPICE 内核全集。
#
# 用途：e2m2e 运行期依赖的 SPICE 内核（星历/闰秒/常数/方向/帧）在项目
# kernels-v1 release 里有一份（GitHub 可达），本脚本提供 NAIF 官方源的一份
# 原始数据（国内网络不可达，Colab 可达），供离线部署/对比/更新使用。
# 文件存到 Google Drive，再从 Drive 取回本地。
#
# 映射关系（NAIF 源 → 项目 kernels/ 用名）：
#   de440s.bsp / de430.bsp             → 同名（行星星历，DE 系列）
#   naif0011.tls / naif0012.tls        → 同名（闰秒）
#   pck00010.tpc                       → 同名（行星常数）
#   earth_latest_high_prec.bpc         → 同名（地球高精度方向）
#   earth_070425_370426_predict.bpc    → SPICEEarthPredictedKernel.bpc（待验证）
#   moon_pa_de421_1900-2050.bpc        → SPICELunaCurrentKernel.bpc（待验证）
#   luna_iau2000.tf                    → SPICELunaFrameKernel.tf（待验证）
#
# 用法：全部复制到 Colab 单元格运行。产物：<Drive>/e2m2e_JPL/（约 170MB）。

import os

import requests
from google.colab import drive

# 1. 挂载 Google Drive
drive.mount("/content/drive")


def download_file(url, filename, save_dir):
    """下载到临时文件再改名落盘：中断的下载不残留半截文件；已存在则跳过。"""
    save_path = os.path.join(save_dir, filename)
    tmp_path = save_path + ".part"
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"已存在，跳过: {filename}")
        return
    print(f"🚀 下载: {filename} ({url})")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        os.replace(tmp_path, save_path)
        size_mb = os.path.getsize(save_path) / 1024 / 1024
        print(f"✅ {filename} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌ {filename}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# 2. Google Drive 存储目录
spice_dir = "/content/drive/MyDrive/e2m2e_JPL"
os.makedirs(spice_dir, exist_ok=True)

# 3. NAIF generic_kernels 下的内核清单：(相对路径, 保存文件名)
NAIF = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
KERNELS = [
    ("spk/planets/de440s.bsp", "de440s.bsp"),
    ("spk/planets/de430.bsp", "de430.bsp"),
    ("lsk/naif0012.tls", "naif0012.tls"),
    ("lsk/naif0011.tls", "naif0011.tls"),
    ("pck/pck00010.tpc", "pck00010.tpc"),
    ("pck/earth_latest_high_prec.bpc", "earth_latest_high_prec.bpc"),
    ("pck/earth_070425_370426_predict.bpc", "SPICEEarthPredictedKernel.bpc"),
    ("pck/moon_pa_de421_1900-2050.bpc", "SPICELunaCurrentKernel.bpc"),
    ("fk/satellites/luna_iau2000.tf", "SPICELunaFrameKernel.tf"),
]

for rel_path, filename in KERNELS:
    download_file(f"{NAIF}/{rel_path}", filename, spice_dir)

print(f"\n完成。目录: {spice_dir}（失败的文件见上方 ❌ 行，可重跑，已下载的会跳过）")
