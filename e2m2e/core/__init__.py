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
"""

from . import (
    coordinate,
    dynamics,
    ephemeris_dynamics,
    ephemeris_system,
    orbit,
    potential,
    spice,
    system,
)
from .coordinate import CoordinateTransformation, ReferenceFrame, SynodicJ2000Transformation
from .dynamics import CR3BP_Dynamics, Dynamics, propagate_state_at_orbit_time
from .ephemeris_dynamics import EphemerisDynamics
from .ephemeris_system import EphemerisSystem
from .orbit import Orbit, OrbitFamily
from .potential import pseudo_potential_hessian
from .spice import SPICEManager
from .srp_dynamics import CR3BP_SRP_Dynamics
from .system import CR3BP_System, LibrationPoint

__all__ = [
    "coordinate",
    "dynamics",
    "potential",
    "ephemeris_dynamics",
    "ephemeris_system",
    "orbit",
    "spice",
    "system",
    "CR3BP_System",
    "LibrationPoint",
    "Dynamics",
    "CR3BP_Dynamics",
    "CR3BP_SRP_Dynamics",
    "Orbit",
    "OrbitFamily",
    "propagate_state_at_orbit_time",
    "CoordinateTransformation",
    "SynodicJ2000Transformation",
    "ReferenceFrame",
    "SPICEManager",
    "EphemerisSystem",
    "EphemerisDynamics",
    "pseudo_potential_hessian",
]
