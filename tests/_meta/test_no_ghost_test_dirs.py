r"""tests/ 目录卫生门禁：不存在只剩 __pycache__ 的幽灵目录。

历次测试重组（ADR 0021/0026）删除测试文件后，残留的空目录（仅含
__pycache__）不进 git，只在本地可见，误导导航。本门禁断言 tests/ 下
不存在这种目录；发现残渣时运行 \`make clean-tests\` 清理。

判据是"递归无非 __pycache__ 文件"而非"无测试文件"：tests/data/types/
fixtures 等夹具目录合法地不含测试文件，不是幽灵。
"""

from __future__ import annotations

import pathlib

import pytest

pytestmark = pytest.mark.aux

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]


def _ghost_dirs(tests_dir: pathlib.Path) -> list[str]:
    """返回仅含 __pycache__ 内容的目录（相对 tests/ 的路径，浅层在前）。"""
    ghosts: list[str] = []
    for directory in sorted(tests_dir.rglob("*")):
        if not directory.is_dir() or directory.name == "__pycache__":
            continue
        real_files = [
            p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        ]
        if not real_files:
            ghosts.append(directory.relative_to(tests_dir).as_posix())
    return ghosts


def test_no_pycache_only_dirs_under_tests():
    ghosts = _ghost_dirs(_TESTS_DIR)
    assert not ghosts, (
        f"tests/ 下存在只剩 __pycache__ 的幽灵目录（{len(ghosts)} 个）：\n"
        + "\n".join(f"  {g}" for g in ghosts)
        + "\n运行 `make clean-tests` 清理后重试。"
    )
