"""力模型类：ForceModel/PhysicalModel 子类/推力。

Python 类是"力模型定义"（参数验证 + to_rust_spec 序列化），与 Rust
``e2m2e-forces`` 的 CompiledForce 枚举对应（ADR 0011，源：``core/forces/``）。
配置 schema 在 ``data/templates/force_config.py`` （纯数据）；大气密度模型
（``atmosphere.py``）一并迁入（源 ``core/atmosphere/``）。

FiniteBurn（恒质量 6D）与 VariableMassFiniteBurn（变质量 7D）均通过 Rust
Rust 编译传播执行；仅可序列化的固定控制可下沉，任意 Python callable 会显式拒绝。
"""

from __future__ import annotations

from .drag import DragModel
from .ecom_srp import EcomSolarRadiationPressure
from .exceptions import (
    CoordinateTransformError,
    NotSerializableError,
    RelativisticCorrectionError,
)
from .force_config import dump_force_config, load_force_config
from .force_model import ForceEntry, ForceModel
from .gravity_field import GravityField
from .indirect_term import IndirectTerm
from .physical_model import PhysicalModel
from .point_mass_gravity import PointMassGravity
from .relativistic_correction import RelativisticCorrection
from .shadow import ConicalShadowModel
from .srp import SolarRadiationPressure, VariableMassSolarRadiationPressure
from .third_body_gravity import ThirdBodyGravity
from .thrust import BurnApplication, FiniteBurn, ImpulsiveBurn, VariableMassFiniteBurn

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
    "VariableMassSolarRadiationPressure",
    "EcomSolarRadiationPressure",
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
