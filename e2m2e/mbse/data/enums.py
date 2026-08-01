"""mbse 数据枚举 shim（ADR 0011 迁移）。

领域枚举定义已迁至 ``e2m2e.data.templates.enums``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.data.templates.enums import (
    BifurcationLabel,
    BoundaryMode,
    ConvergenceState,
    OrbitFamilyType,
    ProjectionPlane,
    ReferenceFrame,
    StabilityLabel,
    TransferPhase,
    TransferType,
    TwoLevelMultipleShootingStatus,
    UnitSystem,
)

__all__ = [
    "BoundaryMode",
    "ConvergenceState",
    "ProjectionPlane",
    "ReferenceFrame",
    "TransferType",
    "TwoLevelMultipleShootingStatus",
    "UnitSystem",
    "OrbitFamilyType",
    "StabilityLabel",
    "BifurcationLabel",
    "TransferPhase",
]
