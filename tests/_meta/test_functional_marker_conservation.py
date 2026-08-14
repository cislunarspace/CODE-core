"""主功能类标记守恒守门员（ADR 0021 决策 1、ADR 0025 决策 2）。

每个已收集用例必须恰好携带一个主功能类标记（theory / integrator / force /
data / orchestration / interface / aux）；``spice``、``low_thrust`` 是正交
标记，不计入主类。ADR 0021 把"每测试恰好 1 主类"写成文字规则后，仍出现
整文件漏标（`test_differential_correction_stagnation.py`）——规则没有
守门员就会漏。本测试即守门员：发现违例即列出 nodeid，倒逼补齐或拆细标记。

注意判读方式：本测试检查的是**本次运行收集到的用例**（session.items）。
完整回归（``make test`` / release 前全量）中它看到全部用例，发挥守门作用；
单独运行本文件时只看到自身（带 ``aux`` 标记），恒绿。
"""

from __future__ import annotations

import pytest

#: ADR 0021 决策 1 的封闭主功能类集合。
PRIMARY_MARKERS = (
    "theory",
    "integrator",
    "force",
    "data",
    "orchestration",
    "interface",
    "aux",
)

pytestmark = pytest.mark.aux


def test_every_collected_test_has_exactly_one_primary_marker(request):
    """每个已收集用例恰好一个主功能类标记；违例列出全部 nodeid。"""
    violations: list[str] = []
    for item in request.session.items:
        present = [name for name in PRIMARY_MARKERS if item.get_closest_marker(name) is not None]
        if len(present) != 1:
            detail = "、".join(present) if present else "（无）"
            violations.append(f"{item.nodeid}  主标记={detail}")
    assert not violations, (
        f"{len(violations)} 个用例违反主标记守恒（ADR 0021：每测试恰好 1 主类；"
        f"spice/low_thrust 为正交标记，不算主类）：\n" + "\n".join(violations)
    )
