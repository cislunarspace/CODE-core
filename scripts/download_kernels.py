"""下载 SPICE 内核文件到 ``kernels/``，供星历动力学测试与运行使用。

与 :mod:`scripts.download_cspice` 对称：后者取 CSPICE 编译包（构建期），
本脚本取星历/姿态/闰秒等内核（运行期）。来源同为项目 GitHub Release
``kernels-v1``，国内网络可达（NAIF 官方源常不可达）。

仓库已提交体积小的内核（.tls/.tpc/.bpc/.tf），仅星历 ``.bsp``（百 MB 级、
``.gitignore`` 忽略）需补。本脚本无差别按扩展名拉取 release 内的全部内核资产，
已存在的文件跳过（幂等），故重复运行零下载。

用法：
    python scripts/download_kernels.py [--kernel-dir DIR]

默认内核目录：仓库根 ``kernels/``（与 ``tests/kernel_helpers.py`` 的
``SPICE_KERNEL_DIR`` 默认一致）。

鉴权：GitHub API 未鉴权限速 60 次/小时，单次 setup 足够；CI 或受限环境设
``GH_TOKEN`` 走鉴权通道（与 download_cspice.py 同）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

REPO = "cislunarspace/e2m2e"
RELEASE = "kernels-v1"
# release.yml 同款 pattern：星历/闰秒/常数/姿态/帧
EXTENSIONS = (".bsp", ".tls", ".tpc", ".bpc", ".tf")

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _api_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = __import__("os").environ.get("GH_TOKEN") or __import__("os").environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _list_release_assets() -> list[dict]:
    """列 ``kernels-v1`` release 的全部资产（name + browser_download_url）。"""
    url = f"https://api.github.com/repos/{REPO}/releases/tags/{RELEASE}"
    req = urllib.request.Request(url, headers=_api_headers())
    with urllib.request.urlopen(req) as resp:  # noqa: S310 — 固定 https API URL
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("assets", [])


def _download(url: str, dest: pathlib.Path) -> None:
    print(f"下载 {url} → {dest}", file=sys.stderr)
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:  # noqa: S310 — 固定 https URL
        fh.write(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 kernels-v1 release 下载 SPICE 内核到 kernels/（幂等）"
    )
    parser.add_argument("--kernel-dir", type=pathlib.Path, default=ROOT / "kernels")
    args = parser.parse_args()

    kernel_dir = args.kernel_dir
    kernel_dir.mkdir(parents=True, exist_ok=True)

    assets = _list_release_assets()
    targets = [a for a in assets if a["name"].lower().endswith(EXTENSIONS)]
    if not targets:
        raise SystemExit(f"release {RELEASE} 未找到内核资产（扩展名 {EXTENSIONS}）")

    fetched = 0
    skipped = 0
    for asset in targets:
        dest = kernel_dir / asset["name"]
        if dest.is_file():
            skipped += 1
            continue
        _download(asset["browser_download_url"], dest)
        fetched += 1

    print(
        f"kernels-v1: 下载 {fetched} 个内核，跳过 {skipped} 个（已存在）→ {kernel_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
