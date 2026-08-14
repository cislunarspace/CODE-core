"""已删目录引用检查（ADR 0021 迁移防漏，#372）。

ADR 0021 五层迁移删了 ``tests/core/``（单数）与 ``tests/algorithms/``（复数）。
迁移完成的标志是源码与测试代码中无任何旧路径引用——``linked_tests``、注释、
docstring 一律不得残留。本脚本扫描 ``e2m2e/`` 与 ``tests/`` 下所有 ``.py``，
发现旧目录路径引用即失败，倒逼彻底清理。

误报处理：ADR / 迁移计划等文档中描述迁移历史的"tests/core"是事实陈述，本脚本
不扫 ``docs/``，自然排除；代码中确属必要的历史标注请改写为不引用旧路径的表述
（如"迁移前位于旧 core 包"）。
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ADR 0021 删除的目录前缀（未来再删别的目录，只改此处）
DELETED_DIRS: tuple[str, ...] = (
    "tests/core",
    "tests/algorithms",
    "tests/numerical/dynamics",
    "tests/data/frames/fixtures",
    "tests/data/atmosphere",
)

# 匹配 "tests/<dir>" 后跟 /、引号、空白或行尾；不匹配 tests/core_xxx 之类的延续
PATTERN = re.compile(r"\b(" + "|".join(re.escape(d) for d in DELETED_DIRS) + r')(?=[/\s\'"]|$)')


def check_file(path: pathlib.Path) -> list[str]:
    """返回该文件违规行列表（``相对路径:行号: 内容``）。"""
    rel = path.relative_to(ROOT)
    violations: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if PATTERN.search(line):
            violations.append(f"{rel}:{lineno}: {line.strip()}")
    return violations


def main() -> int:
    violations: list[str] = []
    for sub in ("e2m2e", "tests"):
        for path in sorted((ROOT / sub).rglob("*.py")):
            violations.extend(check_file(path))
    if violations:
        names = "、".join(DELETED_DIRS)
        print(f"已删目录引用残留（{names} 已被 ADR 0021 删除）：")
        for v in violations:
            print(f"  {v}")
        print("迁移后不得再引用；改写为不引用旧路径的表述，或更新为新路径。")
        return 1
    names = "、".join(DELETED_DIRS)
    print(f"已删目录引用检查通过（{names} 无残留）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
