"""保持点安全分析 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.proximity``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.proximity.safety import (
    KeepingPoint,
    SafetyRegion,
    SafetyReport,
    check_passive_safety,
    max_collision_probability,
)

__all__ = [
    "KeepingPoint",
    "SafetyRegion",
    "SafetyReport",
    "check_passive_safety",
    "max_collision_probability",
]
