"""数据模板：轨道族种子、系统标准参数、摄动开关默认、力模型配置 schema、领域枚举。

- ``seed.py``：轨道族种子参数。
- ``systems.py``：空模块壳，物理常数真值在 ``e2m2e.data.constants``。
- ``perturbations.py``：摄动开关/DYB 默认。
- ``design.py``：design_orbit 星历修正方法的族级分派表。
- ``force_config.py``：力模型配置 schema（纯数据）。
- ``enums.py``：领域枚举。
"""

from __future__ import annotations

from .design import SEGMENTED_CORRECTION_ORBIT_TYPES
from .enums import (
    BifurcationLabel,
    BoundaryMode,
    ConvergenceState,
    FailureCause,
    LibrationPoint,
    OrbitFamilyType,
    ProjectionPlane,
    ReferenceFrame,
    StabilityLabel,
    TransferPhase,
    TransferType,
    UnitSystem,
)
from .perturbations import DEFAULT_DYB, DEFAULT_PERTURBATION
from .seed import (
    CHAR_LENGTH_KM,
    CHAR_PERIOD_SEC,
    EARTH_MOON_MU,
    MOON_RADIUS_KM,
)

__all__ = [
    "CHAR_LENGTH_KM",
    "CHAR_PERIOD_SEC",
    "EARTH_MOON_MU",
    "MOON_RADIUS_KM",
    "DEFAULT_DYB",
    "DEFAULT_PERTURBATION",
    "SEGMENTED_CORRECTION_ORBIT_TYPES",
    "ReferenceFrame",
    "UnitSystem",
    "ProjectionPlane",
    "TransferType",
    "BoundaryMode",
    "ConvergenceState",
    "FailureCause",
    "LibrationPoint",
    "OrbitFamilyType",
    "StabilityLabel",
    "BifurcationLabel",
    "TransferPhase",
]
