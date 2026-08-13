"""下载并解压 SPICE 编译包，输出 CSPICE_DIR 供 cspice-sys 构建使用。

背景：仓库不启用 ``cspice-sys`` 的 ``downloadcspice`` feature（其从 naif.jpl.nasa.gov 下载
CSPICE 源码就地编译，国内网络常不可达，构建时 TCP 超时），CSPICE 一律经本脚本从
e2m2e 的 GitHub Release 下载预编译的 MICE 工具包（release ``cspice-v1`` 的资产
``cspice-windows.zip`` / ``cspice-linux.zip`` / ``cspice-linux-aarch64.zip``），解压后把
``mice_{platform}`` 目录作为 ``CSPICE_DIR`` 输出。``cspice-sys`` 的 build.rs 检测到
``CSPICE_DIR`` 即使用之；缺 ``CSPICE_DIR`` 时构建直接报错（见 cspice-sys build.rs 第 21-33 行）。

资产按「操作系统 × 架构」选择：x86_64 Linux 与 aarch64 Linux 各自独立的 zip
（``cspice-linux.zip`` / ``cspice-linux-aarch64.zip``），aarch64 库由
``.github/workflows/cspice-aarch64-build.yml`` 在 GitHub arm64 runner 上从 NAIF 源码包
编译后发到 ``cspice-v1`` release（NAIF 官方无 Linux ARM64 预编译包）。解压子目录
按资产区分（x86_64 沿用 MICE 包的 ``mice_linux``，aarch64 为 ``mice_linux_aarch64``），
避免同一缓存目录下两架构产物互相覆盖。Windows 仅 x86_64，维持原 ``mice_windows``。

用法：
    python scripts/download_cspice.py [--cache-dir DIR]
    python scripts/download_cspice.py --print-cspice-dir

不传参数：下载（若未缓存）+ 解压到缓存，在 stdout 打印 ``CSPICE_DIR=<dir>``——
正是 GitHub Actions ``>> "$GITHUB_ENV"`` 的变量文件格式（CI 用），也兼容
``eval "$(python3 scripts/download_cspice.py)"`` 的 shell 赋值。
``--print-cspice-dir``：仅打印已解压的 ``CSPICE_DIR`` 纯路径（供 shell ``$(...)`` 捕获）。

默认缓存目录：仓库根 ``.cspice/``（gitignore 已忽略）。
"""

from __future__ import annotations

import argparse
import pathlib
import platform
import shutil
import sys
import urllib.request
import zipfile

REPO = "cislunarspace/e2m2e"
RELEASE = "cspice-v1"
# (操作系统, 架构) → (release 资产名, 解压后 CSPICE 子目录名)
ASSET_BY_PLATFORM = {
    ("windows", "x86_64"): ("cspice-windows.zip", "mice_windows"),
    ("linux", "x86_64"): ("cspice-linux.zip", "mice_linux"),
    ("linux", "aarch64"): ("cspice-linux-aarch64.zip", "mice_linux_aarch64"),
}
# 架构归一化：各平台对同一 ISA 的 machine 字符串不尽相同
_MACHINE_ALIASES = {"amd64": "x86_64", "arm64": "aarch64"}

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _platform() -> tuple[str, str]:
    """返回 (操作系统, 架构)，二者都归一化到 ASSET_BY_PLATFORM 的键。"""
    sys_platform = sys.platform
    if sys_platform.startswith("win"):
        os_name = "windows"
    elif sys_platform.startswith("linux"):
        os_name = "linux"
    else:
        raise SystemExit(f"不支持平台：{sys_platform}（当前仅支持 windows/linux）")
    machine = _MACHINE_ALIASES.get(platform.machine().lower(), platform.machine().lower())
    return os_name, machine


def _asset(platform_key: tuple[str, str]) -> tuple[str, str]:
    try:
        return ASSET_BY_PLATFORM[platform_key]
    except KeyError:
        os_name, machine = platform_key
        raise SystemExit(
            f"暂无 {os_name}/{machine} 的 CSPICE 编译包：资产表 {sorted(ASSET_BY_PLATFORM)}"
        ) from None


def _asset_url(asset: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{RELEASE}/{asset}"


def _ensure_extracted(cache_dir: pathlib.Path) -> pathlib.Path:
    """下载（若未缓存）并解压 MICE 包，返回指向 ``mice_{platform}`` 的目录。"""
    asset, subdir = _asset(_platform())
    target = cache_dir / subdir
    if target.is_dir() and (target / "include" / "SpiceUsr.h").is_file():
        return target

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / asset
    if not zip_path.is_file():
        url = _asset_url(asset)
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
    if _platform()[0] == "linux":
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
    # 输出 GITHUB_ENV 变量文件格式（``NAME=VALUE``）：CI 用
    # ``>> "$GITHUB_ENV"`` 落盘设置 CSPICE_DIR，shell 侧 ``eval $(...)`` 同样可用。
    print(f"CSPICE_DIR={cspice_dir}")


if __name__ == "__main__":
    main()
