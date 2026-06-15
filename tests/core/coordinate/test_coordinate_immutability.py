"""守门员测试:扫描 e2m2e/core/ 内不应有 Axes / Origin 属性的运行时 mutate。

issue #76 验收第 5 条"代码遵循 immutability 与类型注解规范"的字面落实——
本文件只约束 core 层(具体 Axes / Origin 子类的实现细节归 #121 全量冻结 issue
与 #122 DynamicAxes 边界 issue,超出 #76 范围)。CoordinateSystem 的冻结契约
在 test_coordinate_system.py::TestCoordinateSystemFrozen 里验证。
"""

from __future__ import annotations

import re
from pathlib import Path


def test_no_axes_or_origin_attribute_mutation_in_core():
    """e2m2e/core/ 内不应出现 axes.attr = ... / origin.attr = ... 等 mutate。

    扫描范围:仅 `e2m2e/core/`(纯核心层);其他上层模块(forces / transfer /
    algorithms)允许出现局部变量名为 `axes` / `origin` 的常规用法,本规则不
    覆盖。如有误报可缩小匹配窗口。
    """
    core_root = Path(__file__).resolve().parents[3] / "e2m2e" / "core"
    assert core_root.is_dir(), f"core 目录不存在:{core_root}"

    # 匹配 cs.axes = ...  system.axes = ...  coord.axes = ... 这类
    # "明确标识符前缀" 的属性赋值;排除 cs = ... axes = ... (普通变量赋值)
    pattern = re.compile(
        r"\b(?:cs|coord|system|target|source|itrf|icrf|to_cs|from_cs)"
        r"\.(?:axes|origin)\s*="
    )

    offenders: list[tuple[str, int, str]] = []
    for py in sorted(core_root.rglob("*.py")):
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(
                    (str(py.relative_to(core_root)), lineno, line.strip())
                )

    assert not offenders, (
        "e2m2e/core/ 检测到 Axes/Origin 属性的运行时 mutate,违反 immutability 规范:\n"
        + "\n".join(f"  {p}:{n}: {txt}" for p, n, txt in offenders)
    )
