"""定位 Windows Rust 测试所需的 Python DLL 目录，供 Makefile 注入 PATH。

``cargo test`` 不启用 PyO3 ``extension-module``：测试 EXE 会链接 Python（当前 ABI3 链接
``python3.dll``），但该 DLL 不在测试 EXE 目录，需要由 PATH 提供给 Windows loader。
虚拟环境的 ``Scripts`` 目录不含该 DLL，实际安装目录见 ``sys.base_prefix``（issue #495）。

用法：
    python scripts/python_dll_dir.py [--override DIR]

正常时在 stdout 打印目录纯路径（供 ``$(...)`` 捕获）；找不到任何 ``python*.dll`` 时
向 stderr 报告并返回非零。
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def _dlls_in(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(directory.glob("python*.dll"))


def find_python_dll_dir(override: str | None) -> pathlib.Path:
    """返回含 python*.dll 的目录；override 优先，其次 sys.base_prefix。"""
    candidates: list[tuple[str, pathlib.Path]] = []
    if override is not None:
        candidates.append(("--override", pathlib.Path(override).expanduser()))
    else:
        candidates.append(("sys.base_prefix", pathlib.Path(sys.base_prefix)))
        candidates.append(("sys.executable 所在目录", pathlib.Path(sys.executable).parent))

    for _source, directory in candidates:
        if directory.is_dir() and _dlls_in(directory):
            return directory.resolve()

    detail = "、".join(f"{source}（{directory}）" for source, directory in candidates)
    raise SystemExit(
        f"找不到 Python DLL：已检查 {detail}，均无 python*.dll。"
        "请用 --override 或 make PYTHON_DLL_DIR=... 指定含 python*.dll 的目录。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--override", help="直接使用该目录，仍验证其中含 python*.dll")
    args = parser.parse_args()
    print(find_python_dll_dir(args.override))


if __name__ == "__main__":
    main()
