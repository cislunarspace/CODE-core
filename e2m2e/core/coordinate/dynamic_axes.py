"""动态坐标轴 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.coordinate``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.coordinate import DynamicAxes

__all__ = ["DynamicAxes"]
