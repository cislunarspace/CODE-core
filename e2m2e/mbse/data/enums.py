"""mbse 数据枚举

基础领域枚举（``ReferenceFrame``、``UnitSystem``、``ProjectionPlane``、
``TransferType``、``BoundaryMode``、``TwoLevelMultipleShootingStatus``、
``ConvergenceState``）已下沉到 ``e2m2e.core.enums``，此处仅作重导出以保持向后兼容。

仅 mbse 内部使用的追溯性枚举（``OrbitFamilyType``、``StabilityLabel``、
``BifurcationLabel``、``TransferPhase``）仍定义在本模块。
"""

from __future__ import annotations

import enum

from e2m2e.core.enums import (
    BoundaryMode,
    ConvergenceState,
    ProjectionPlane,
    ReferenceFrame,
    TransferType,
    TwoLevelMultipleShootingStatus,
    UnitSystem,
)

__all__ = [
    # 自 core 重导出的基础领域枚举
    "BoundaryMode",
    "ConvergenceState",
    "ProjectionPlane",
    "ReferenceFrame",
    "TransferType",
    "TwoLevelMultipleShootingStatus",
    "UnitSystem",
    # mbse 内部追溯性枚举
    "OrbitFamilyType",
    "StabilityLabel",
    "BifurcationLabel",
    "TransferPhase",
]


class OrbitFamilyType(enum.Enum):
    """轨道族类型"""

    HALO = "halo"
    LYAPUNOV = "lyapunov"
    VERTICAL = "vertical"
    AXIAL = "axial"
    BUTTERFLY = "butterfly"
    DRAGONFLY = "dragonfly"
    DRO = "dro"  # Distant Retrograde Orbit
    DPO = "dpo"  # Direct Prograde Orbit (待实现)
    SPO = "spo"  # Short Period Orbit (待实现)
    LPO = "lpo"  # Long Period Orbit (待实现)
    TADPOLE = "tadpole"  # 待实现
    HORSESHOE = "horseshoe"  # 待实现
    RO = "ro"  # Resonant Orbit
    NRHO = "nrho"  # Near Rectilinear Halo Orbit
    LYO = "lyo"  # Lissajous Orbit


class StabilityLabel(enum.Enum):
    """轨道稳定性标签"""

    STABLE = "stable"
    UNSTABLE = "unstable"
    MARGINALLY_STABLE = "marginally_stable"
    HYPERBOLIC = "hyperbolic"
    ELLIPTIC = "elliptic"
    PARABOLIC = "parabolic"


class BifurcationLabel(enum.Enum):
    """分岔类型标签"""

    NONE = "none"
    PERIOD_DOUBLING = "period_doubling"
    SADDLE_NODE = "saddle_node"
    TORUS = "torus"
    PITCHFORK = "pitchfork"
    TRANSCRITICAL = "transcritical"
    SECONDARY_HOPF = "secondary_hopf"


class TransferPhase(enum.Enum):
    """转移设计阶段（用于状态机图）"""

    CONFIGURED = "configured"
    SEARCHING = "searching"
    CANDIDATES_FOUND = "candidates_found"
    OPTIMIZING = "optimizing"
    COMPLETE = "complete"
    FAILED = "failed"
