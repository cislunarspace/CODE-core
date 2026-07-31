"""PhysicalModel 抽象基类 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.forces``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.forces.physical_model import (
    PhysicalModel,
    require_inertial_frame,
)

__all__ = ["PhysicalModel", "require_inertial_frame"]
