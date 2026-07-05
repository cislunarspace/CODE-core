"""e2m2e 核心模块

包含三体问题系统定义、动力学方程、轨道数据结构、坐标系与轴/原点组件。

Classes:
    CR3BP_System: 圆型限制性三体问题系统
    LibrationPoint: 平动点枚举
    Dynamics: 通用动力学基类
    CR3BP_Dynamics: CR3BP 动力学方程
    Orbit: 轨道数据容器
    OrbitFamily: 轨道族容器
    ReferenceFrame: 参考坐标系枚举
    SynodicJ2000System: 基于 CoordinateSystem 的 synodic ↔ J2000 转换器

Functions:
    propagate_state_at_orbit_time: 沿轨道周期外推状态

Note:
    星历相关模块 (spice, ephemeris_system, ephemeris_dynamics) 不在顶层导出，
    如需使用请直接从子模块导入，例如
    ``from e2m2e.core.spice import SPICEManager``。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ephemeris_dynamics import EphemerisDynamics as EphemerisDynamics
    from .ephemeris_system import EphemerisSystem as EphemerisSystem
    from .spice import SPICEManager as SPICEManager
    from .standard_axes import ITRFSpiceAxes as ITRFSpiceAxes
    from .standard_origins import CelestialBodyOrigin as CelestialBodyOrigin

from . import dynamics, orbit, potential, system
from .axes import Axes
from .coordinate_system import CoordinateSystem
from .cr3bp_system import CR3BP_System, LibrationPoint
from .dynamic_axes import DynamicAxes
from .dynamics import CR3BP_Dynamics, Dynamics, propagate_state_at_orbit_time
from .enums import ReferenceFrame
from .orbit import Orbit, OrbitFamily
from .origin import Origin
from .potential import pseudo_potential_hessian
from .standard_axes import (
    GMATITRFAxes,
    IAU2000EqAxes,
    ICRSAxes,
    ITRFApproxAxes,
    ITRFAxes,
    standard_icrf,
    standard_itrf,
)
from .standard_dynamic_axes import LVLHAxes, VNBAxes
from .standard_origins import InertialOrigin
from .synodic_j2000 import SynodicJ2000System
from .system import System

# 星历/SPICE 相关符号通过 __getattr__ 按需延迟导入，避免用户只使用 CR3BP
# 基础类时强制加载 spiceypy。参见 issue #44。
_LAZY_SPICE_EXPORTS: dict[str, str] = {
    "SPICEManager": "e2m2e.core.spice",
    "EphemerisSystem": "e2m2e.core.ephemeris_system",
    "EphemerisDynamics": "e2m2e.core.ephemeris_dynamics",
    "ITRFSpiceAxes": "e2m2e.core.standard_axes",
    "CelestialBodyOrigin": "e2m2e.core.standard_origins",
}


def __getattr__(name: str) -> object:
    """按需延迟导入 SPICE/星历相关符号。"""
    module_name = _LAZY_SPICE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'e2m2e.core' has no attribute '{name}'")

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """确保 dir(e2m2e.core) 包含延迟导出的公开符号。"""
    return sorted(set(__all__) | set(globals().keys()) | set(_LAZY_SPICE_EXPORTS.keys()))


__all__ = [
    "dynamics",
    "potential",
    "orbit",
    "system",
    "System",
    "Axes",
    "Origin",
    "CoordinateSystem",
    "ICRSAxes",
    "IAU2000EqAxes",
    "ITRFSpiceAxes",
    "ITRFAxes",
    "GMATITRFAxes",
    "ITRFApproxAxes",
    "standard_icrf",
    "standard_itrf",
    "DynamicAxes",
    "LVLHAxes",
    "VNBAxes",
    "CelestialBodyOrigin",
    "InertialOrigin",
    "SPICEManager",
    "CR3BP_System",
    "LibrationPoint",
    "Dynamics",
    "CR3BP_Dynamics",
    "Orbit",
    "OrbitFamily",
    "propagate_state_at_orbit_time",
    "ReferenceFrame",
    "SynodicJ2000System",
    "pseudo_potential_hessian",
]
