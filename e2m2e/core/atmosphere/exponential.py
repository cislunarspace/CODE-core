"""标准指数大气密度模型 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.forces.atmosphere``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere

__all__ = ["ExponentialAtmosphere"]
