"""守门员测试:扫描 e2m2e/core/ 内不应有 Axes / Origin 属性的运行时 mutate。

这是**代码风格**层面的静态守门员:防止作者在 core 层主动写
``cs.axes = X`` / ``system.origin = Y`` 这类赋值。它不提供运行时偷换防护
(那本就不在应用层职责内),与 ``CoordinateSystem`` 是否 ``frozen`` 无关。

历史背景:#76 曾把 ``CoordinateSystem`` 冻结为 ``@dataclass(frozen=True)``,
后经重新讨论判定冻结不必要(篡改防护由 GitHub 代码来源验证承担,YAGNI),
冻结已回退,#121/#122 关闭 wontfix。本守门员保留——它防的是"作者写 mutate"
而非"运行时偷换",与冻结无关。
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
                offenders.append((str(py.relative_to(core_root)), lineno, line.strip()))

    assert not offenders, (
        "e2m2e/core/ 检测到 Axes/Origin 属性的运行时 mutate,违反 immutability 规范:\n"
        + "\n".join(f"  {p}:{n}: {txt}" for p, n, txt in offenders)
    )
