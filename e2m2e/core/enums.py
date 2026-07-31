"""共享领域枚举 shim（ADR 0011 迁移）。

枚举定义已迁至 ``e2m2e.data.templates.enums``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.data.templates.enums import (
    BoundaryMode,
    ConvergenceState,
    ProjectionPlane,
    ReferenceFrame,
    TransferType,
    TwoLevelMultipleShootingStatus,
    UnitSystem,
)

__all__ = [
    "ReferenceFrame",
    "UnitSystem",
    "ProjectionPlane",
    "TransferType",
    "BoundaryMode",
    "TwoLevelMultipleShootingStatus",
    "ConvergenceState",
]
