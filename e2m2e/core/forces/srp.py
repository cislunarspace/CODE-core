"""太阳光压力模型 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.forces``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.forces import SolarRadiationPressure

__all__ = ["SolarRadiationPressure"]
