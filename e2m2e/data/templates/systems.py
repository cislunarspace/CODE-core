"""地月系统标准参数：μ、特征尺度、物理常量。

数据模板（纯数据）；模型类（``System``/``CR3BP_System``）在 ``algorithm/dynamics/``
（ADR 0011）。

实现状态：骨架。常量待从 ``core/constants.py`` 与 ``core/cr3bp_system.py`` 迁入。
"""

from __future__ import annotations

__all__: list[str] = []

# 以下常量待迁入（当前在 core/constants.py）：
#   G、AU、DAY、YEAR 等物理常量
