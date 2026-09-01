r"""清理 tests/ 下的幽灵目录与字节码缓存（#603）。

历次测试重组删除测试文件后，本地残留只剩 __pycache__ 的空目录（不进
git）。本脚本做两件事：

1. 删除 tests/ 下所有 ``__pycache__`` 目录；
2. 删除之后递归内容为空的目录（浅层在后，父目录随子目录清空而移除）。

默认只报告；``--fix`` 才真正删除。与 tests/_meta/test_no_ghost_test_dirs.py
的判据一致（递归无非 __pycache__ 文件即幽灵），互为独立实现：门禁不能与
清洁工具共享同一段可能有 bug 的逻辑。

用法：\`make clean-tests\`（即 \`python scripts/clean_test_residue.py --fix\`）。
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

TESTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "tests"


def find_pycache_dirs(tests_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(tests_dir.rglob("__pycache__"))


def find_ghost_dirs(tests_dir: pathlib.Path) -> list[pathlib.Path]:
    """仅含 __pycache__ 内容的目录（含彼此嵌套的父目录），浅层在前。"""
    ghosts: list[pathlib.Path] = []
    for directory in sorted(tests_dir.rglob("*")):
        if not directory.is_dir() or directory.name == "__pycache__":
            continue
        if not any(p.is_file() and "__pycache__" not in p.parts for p in directory.rglob("*")):
            ghosts.append(directory)
    return ghosts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fix", action="store_true", help="真正删除；缺省只报告")
    options = parser.parse_args()

    pycache = find_pycache_dirs(TESTS_DIR)
    ghosts = find_ghost_dirs(TESTS_DIR)
    if not pycache and not ghosts:
        print("tests/ 干净：无 __pycache__，无幽灵目录")
        return 0

    for directory in pycache:
        print(f"__pycache__: {directory.relative_to(TESTS_DIR.parent)}")
    for directory in ghosts:
        print(f"幽灵目录:   {directory.relative_to(TESTS_DIR.parent)}")
    if not options.fix:
        print(f"\n共 {len(pycache)} 个 __pycache__、{len(ghosts)} 个幽灵目录（未删除；加 --fix）")
        return 0

    for directory in pycache:
        shutil.rmtree(directory, ignore_errors=True)
    # 浅层在前：子目录清空后父目录才能被判空删除（rmtree 对非空不报错地
    # 失败会留残渣，故按 find_ghost_dirs 的浅层序整树删除）。
    for directory in ghosts:
        shutil.rmtree(directory, ignore_errors=True)
    leftover = find_ghost_dirs(TESTS_DIR)
    if leftover:
        names = "、".join(str(d) for d in leftover)
        print(f"清理后仍残留：{names}", file=sys.stderr)
        return 1
    print(f"已清理 {len(pycache)} 个 __pycache__、{len(ghosts)} 个幽灵目录")
    return 0


if __name__ == "__main__":
    sys.exit(main())
