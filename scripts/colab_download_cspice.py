# 在 Google Colab 中运行的下载脚本：从 NAIF 官网下载 CSPICE 源码包。
#
# 用途：e2m2e 支持 aarch64 Linux 需要 aarch64 的 libcspice.a（NAIF 官方没有
# Linux ARM64 预编译包，必须拿源码在 arm64 上 make）。NAIF 官网国内网络不可达，
# 但 Colab 可访问（用户已实测成功下载 Voyager 内核）。本脚本把源码包存到
# Google Drive，再从 Drive 取回本地（.cspice/ 或任意目录）。
#
# 下载源与 cspice-sys 1.0.4 的 downloadcspice feature 同 URL：
#   https://naif.jpl.nasa.gov/pub/naif/toolkit//C/PC_Linux_GCC_64bit/packages/cspice.tar.Z
# 该包是平台源码发行包（含 makefile），架构无关，在 aarch64 上 make 即得
# lib/cspice.a（重命名 libcspice.a 后即 cspice-sys 需要的静态库）。
#
# 用法：全部复制到 Colab 单元格运行。产物：<Drive>/e2m2e_CSPICE/cspice.tar.Z（约 60MB）。

import ipaddress
import os
import socket
import urllib.parse
from pathlib import Path

import requests
from google.colab import drive

# 1. 挂载 Google Drive
drive.mount("/content/drive")


def assert_safe_url(url):
    """服务端请求边界：仅 http/https，且解析后全部地址为公网。

    拒绝 localhost、环回、私有与保留段（含云元数据端点）。校验通过
    返回原 url，供调用点内联使用。
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"仅允许 http/https 协议：{url}")
    host = parts.hostname or ""
    for info in socket.getaddrinfo(host, None):
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ValueError(f"禁止访问非公网地址：{host}（{ip}）")
    return url


def download_file(url, filename, save_dir):
    """下载到临时文件再改名落盘：中断的下载不残留半截文件。"""
    save_path = os.path.join(save_dir, filename)
    tmp_path = save_path + ".part"
    if os.path.exists(save_path):
        size_mb = os.path.getsize(save_path) / 1024 / 1024
        print(f"文件已存在，跳过下载: {filename} ({size_mb:.1f} MB)")
        return
    print(f"🚀 正在下载: {filename}...")
    try:
        # allow_redirects=False：重定向目标逐一过 assert_safe_url，防止
        # 302 跳向内网/元数据端点（初始 URL 合法不代表跳转目标合法）。
        response = requests.get(
            assert_safe_url(url), stream=True, timeout=60, allow_redirects=False
        )
        hops = 0
        while response.is_redirect or response.is_permanent_redirect:
            hops += 1
            if hops > 5:
                raise ValueError(f"重定向次数过多：{url}")
            response = requests.get(
                assert_safe_url(response.headers["Location"]),
                stream=True,
                timeout=60,
                allow_redirects=False,
            )
        response.raise_for_status()
        with Path(tmp_path).open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        os.replace(tmp_path, save_path)
        size_mb = os.path.getsize(save_path) / 1024 / 1024
        print(f"✅ 下载完成: {filename} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌ 下载失败 {filename}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# 2. Google Drive 存储目录
spice_dir = "/content/drive/MyDrive/e2m2e_CSPICE"
os.makedirs(spice_dir, exist_ok=True)

# 3. CSPICE 源码包（PC_Linux_GCC_64bit：与 cspice-sys downloadcspice 同源，
#    编译目标架构由编译机器的 makefile 决定，aarch64 上 make 即得 arm64 库）
url = "https://naif.jpl.nasa.gov/pub/naif/toolkit//C/PC_Linux_GCC_64bit/packages/cspice.tar.Z"
download_file(url, "cspice.tar.Z", spice_dir)

print(f"\n🎉 已保存: {spice_dir}/cspice.tar.Z")
