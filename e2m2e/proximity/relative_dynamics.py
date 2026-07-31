"""相对运动动力学 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.proximity``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.proximity.relative_dynamics import (
    RelativeDynamics,
    RelativeState,
    TargetOrbit,
)

__all__ = ["RelativeDynamics", "RelativeState", "TargetOrbit"]
