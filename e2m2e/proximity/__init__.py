"""交会对接与相对运动模块 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.proximity``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.proximity import (
    KeepingPoint,
    PhasingManeuver,
    PhasingSolution,
    RelativeDynamics,
    RelativeState,
    SafetyRegion,
    SafetyReport,
    TargetOrbit,
    check_passive_safety,
    max_collision_probability,
    phasing_search,
)

__all__ = [
    "KeepingPoint",
    "PhasingManeuver",
    "PhasingSolution",
    "RelativeDynamics",
    "RelativeState",
    "SafetyRegion",
    "SafetyReport",
    "TargetOrbit",
    "check_passive_safety",
    "max_collision_probability",
    "phasing_search",
]
