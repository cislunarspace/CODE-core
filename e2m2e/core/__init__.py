"""e2m2e 核心模块

包含三体问题系统定义、动力学方程、轨道数据结构和坐标变换的核心类。

Classes:
    CR3BP_System: 圆型限制性三体问题系统
    LibrationPoint: 平动点枚举
    Dynamics: 通用动力学基类
    CR3BP_Dynamics: CR3BP 动力学方程
    Orbit: 轨道数据容器
    OrbitFamily: 轨道族容器
    CoordinateTransformation: 坐标系变换
    ReferenceFrame: 参考坐标系枚举

Functions:
    propagate_state_at_orbit_time: 沿轨道周期外推状态

Note:
    星历相关模块 (spice, ephemeris_system, ephemeris_dynamics) 不在顶层导出，
    如需使用请直接从子模块导入：
        from e2m2e.core.spice import SPICEManager
        from e2m2e.core.ephemeris_system import EphemerisSystem
        from e2m2e.core.ephemeris_dynamics import EphemerisDynamics
"""

from __future__ import annotations

from . import coordinate, dynamics, orbit, potential, system
from .axes import Axes
from .coordinate import CoordinateTransformation, ReferenceFrame, SynodicJ2000Transformation
from .coordinate_system import CoordinateSystem
from .cr3bp_system import CR3BP_System, LibrationPoint
from .dynamics import CR3BP_Dynamics, Dynamics, propagate_state_at_orbit_time
from .orbit import Orbit, OrbitFamily
from .origin import Origin
from .potential import pseudo_potential_hessian
from .spice import SPICEManager
from .standard_axes import (
    GMATITRFAxes,
    IAU2000EqAxes,
    ICRSAxes,
    ITRFApproxAxes,
    ITRFAxes,
    ITRFSpiceAxes,
    standard_itrf,
)
from .standard_origins import CelestialBodyOrigin, InertialOrigin
from .system import System

__all__ = [
    "coordinate",
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
    "standard_itrf",
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
    "CoordinateTransformation",
    "SynodicJ2000Transformation",
    "ReferenceFrame",
    "pseudo_potential_hessian",
]
