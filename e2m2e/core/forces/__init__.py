"""e2m2e 力模型子包 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.forces``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.forces import (
    BurnApplication,
    ConicalShadowModel,
    CoordinateTransformError,
    DragModel,
    FiniteBurn,
    ForceEntry,
    ForceModel,
    GravityField,
    ImpulsiveBurn,
    IndirectTerm,
    NotSerializableError,
    PhysicalModel,
    PointMassGravity,
    RelativisticCorrection,
    RelativisticCorrectionError,
    SolarRadiationPressure,
    ThirdBodyGravity,
    VariableMassFiniteBurn,
    dump_force_config,
    load_force_config,
)

__all__ = [
    "PhysicalModel",
    "PointMassGravity",
    "ThirdBodyGravity",
    "ForceModel",
    "ForceEntry",
    "GravityField",
    "IndirectTerm",
    "DragModel",
    "SolarRadiationPressure",
    "ConicalShadowModel",
    "ImpulsiveBurn",
    "FiniteBurn",
    "VariableMassFiniteBurn",
    "BurnApplication",
    "RelativisticCorrection",
    "RelativisticCorrectionError",
    "CoordinateTransformError",
    "NotSerializableError",
    "load_force_config",
    "dump_force_config",
]
