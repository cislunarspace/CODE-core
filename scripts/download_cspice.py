"""下载并解压 SPICE 编译包，输出 CSPICE_DIR 供 cspice-sys 构建使用。

背景：``cspice-sys`` 默认的 ``downloadcspice`` 从 naif.jpl.nasa.gov 下载 CSPICE
源码就地编译，国内网络常不可达（构建时 TCP 超时）。本脚本改从 e2m2e 的
GitHub Release 下载预编译的 MICE 工具包（``cspice-windows-v1`` /
``cspice-linux-v1``），解压后把 ``mice_{platform}`` 目录作为 ``CSPICE_DIR``
输出。``cspice-sys`` 的 build.rs 检测到 ``CSPICE_DIR`` 即跳过 NAIF 下载
（见 cspice-sys build.rs 第 21-33 行）。

用法：
    python scripts/download_cspice.py [--cache-dir DIR]
    python scripts/download_cspice.py --print-cspice-dir

不传参数：下载（若未缓存）+ 解压到缓存，在 stdout 打印 ``CSPICE_DIR=<dir>``。
``--print-cspice-dir``：仅打印已解压的 ``CSPICE_DIR``（供 shell ``$(...)`` 捕获）。

默认缓存目录：仓库根 ``.cspice/``（gitignore 已忽略）。
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import urllib.request
import zipfile

REPO = "cislunarspace/e2m2e"
RELEASE = "cspice-v1"
ASSET_BY_PLATFORM = {
    "windows": "cspice-windows.zip",
    "linux": "cspice-linux.zip",
}
# MICE 包内，include/ 与 lib/ 位于 mice_{platform}/ 子目录
CSPICE_SUBDIR = {"windows": "mice_windows", "linux": "mice_linux"}

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _platform() -> str:
    sys_platform = sys.platform
    if sys_platform.startswith("win"):
        return "windows"
    if sys_platform.startswith("linux"):
        return "linux"
    raise SystemExit(f"不支持平台：{sys_platform}（当前仅支持 windows/linux）")


def _asset_url(platform_name: str) -> str:
    asset = ASSET_BY_PLATFORM[platform_name]
    return f"https://github.com/{REPO}/releases/download/{RELEASE}/{asset}"


def _ensure_extracted(cache_dir: pathlib.Path) -> pathlib.Path:
    """下载（若未缓存）并解压 MICE 包，返回指向 ``mice_{platform}`` 的目录。"""
    platform_name = _platform()
    subdir = CSPICE_SUBDIR[platform_name]
    target = cache_dir / subdir
    if target.is_dir() and (target / "include" / "SpiceUsr.h").is_file():
        return target

    cache_dir.mkdir(parents=True, exist_ok=True)
    asset = ASSET_BY_PLATFORM[platform_name]
    zip_path = cache_dir / asset
    if not zip_path.is_file():
        url = _asset_url(platform_name)
        print(f"下载 {url} → {zip_path}", file=sys.stderr)
        urllib.request.urlretrieve(url, zip_path)  # noqa: S310 — 固定 https URL
    print(f"解压 {zip_path} → {cache_dir}", file=sys.stderr)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    if not target.is_dir():
        raise SystemExit(f"解压后未找到 {subdir}/（zip 结构异常）")
    # 平台修补：cspice-sys 用 ``rustc-link-lib=static=cspice`` 链接，
    # Linux 下 cargo 找 ``libcspice.a``，但 MICE 包 lib/ 只带 ``cspice.a``
    # （官方 downloadcspice 分支会重命名，CSPICE_DIR 路径不会）。
    if platform_name == "linux":
        src = target / "lib" / "cspice.a"
        dst = target / "lib" / "libcspice.a"
        if src.is_file() and not dst.is_file():
            shutil.copyfile(src, dst)
            print(f"已生成 {dst}（cspice-sys 需要 libcspice.a）", file=sys.stderr)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="下载并解压 SPICE 编译包，输出 CSPICE_DIR（供 cspice-sys 构建）"
    )
    parser.add_argument("--cache-dir", type=pathlib.Path, default=ROOT / ".cspice")
    parser.add_argument(
        "--print-cspice-dir",
        action="store_true",
        help="仅打印已解压的 CSPICE_DIR（供 shell $(...) 捕获）",
    )
    args = parser.parse_args()

    cspice_dir = _ensure_extracted(args.cache_dir)
    if args.print_cspice_dir:
        print(cspice_dir)
        return
    print(f"CSPICE_DIR={cspice_dir}")


if __name__ == "__main__":
    main()
