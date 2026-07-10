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
    from .coordinate.standard_axes import ITRFSpiceAxes as ITRFSpiceAxes
    from .coordinate.standard_origins import CelestialBodyOrigin as CelestialBodyOrigin
    from .ephemeris_system import EphemerisSystem as EphemerisSystem
    from .spice import SPICEManager as SPICEManager

from . import dynamics, orbit, potential, system
from .coordinate import (
    Axes,
    CoordinateSystem,
    DynamicAxes,
    IAU2000EqAxes,
    GMATITRFAxes,
    ICRSAxes,
    ITRFApproxAxes,
    ITRFAxes,
    InertialOrigin,
    LVLHAxes,
    Origin,
    VNBAxes,
    standard_icrf,
    standard_itrf,
)
from .coordinate import SynodicJ2000System
from .cr3bp_system import CR3BP_System, LibrationPoint
from .dynamics import CR3BP_Dynamics, Dynamics, propagate_state_at_orbit_time
from .enums import ReferenceFrame
from .orbit import Orbit, OrbitFamily
from .potential import pseudo_potential_hessian
from .system import System

# 坐标系族已收入 .coordinate 子包（issue #197）。为保持对外导入路径零破坏，
# 把旧路径 e2m2e.core.<mod> 注册为指向 e2m2e.core.coordinate.<mod> 的别名，
# 使 `from e2m2e.core.axes import Axes` 等既有写法继续可用。
import sys as _sys

from . import coordinate as _coordinate

for _mod in (
    "axes",
    "origin",
    "coordinate_system",
    "dynamic_axes",
    "standard_axes",
    "standard_origins",
    "standard_dynamic_axes",
    "synodic_axes",
    "synodic_j2000",
    "rho_bridge",
    "xys",
    "iau_2006",
):
    _full = f"{__name__}.{_mod}"
    if _full not in _sys.modules:
        _sys.modules[_full] = importlib.import_module(f"{__name__}.coordinate.{_mod}")
del _sys, _coordinate, _mod, _full

# 星历/SPICE 相关符号通过 __getattr__ 按需延迟导入，避免用户只使用 CR3BP
# 基础类时强制加载 spiceypy。参见 issue #44。
_LAZY_SPICE_EXPORTS: dict[str, str] = {
    "SPICEManager": "e2m2e.core.spice",
    "EphemerisSystem": "e2m2e.core.ephemeris_system",
    "ITRFSpiceAxes": "e2m2e.core.coordinate.standard_axes",
    "CelestialBodyOrigin": "e2m2e.core.coordinate.standard_origins",
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
