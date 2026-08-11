"""数据模板：轨道族种子、系统标准参数、摄动开关默认、力模型配置 schema、领域枚举。

实现状态：已迁移（ADR 0011 第 1 批）。

- ``seed.py``：轨道族种子参数（源 ``dfh/cr3bp_orbits.py`` 常量）。
- ``systems.py``：物理常量与系统标准参数（源 ``core/constants.py`` +
  ``core/cr3bp_system.py`` 参数）。
- ``perturbations.py``：摄动开关/DYB 默认（源 ``io/inputs_dac.py`` DEFAULT_*）。
- ``force_config.py``：力模型配置 schema（纯数据，源 ``core/forces/``）。
- ``enums.py``：领域枚举（源 ``core/enums.py`` + ``mbse/data/enums.py``）。
"""

from __future__ import annotations

from .enums import (
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
    "ReferenceFrame",
    "UnitSystem",
    "ProjectionPlane",
    "TransferType",
    "BoundaryMode",
    "TwoLevelMultipleShootingStatus",
    "ConvergenceState",
    "OrbitFamilyType",
    "StabilityLabel",
    "BifurcationLabel",
    "TransferPhase",
]
