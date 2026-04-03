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

Functions:
    propagate_state_at_orbit_time: 沿轨道周期外推状态
"""

from . import system
from . import dynamics
from . import orbit
from . import coordinate
from . import spice
from . import ephemeris_system
from . import ephemeris_dynamics
from . import synodic_j2000
from . import homotopy_dynamics

from .system import CR3BP_System, LibrationPoint
from .dynamics import Dynamics, CR3BP_Dynamics, propagate_state_at_orbit_time
from .orbit import Orbit, OrbitFamily
from .coordinate import CoordinateTransformation
from .spice import SPICEManager
from .ephemeris_system import EphemerisSystem
from .ephemeris_dynamics import EphemerisDynamics
from .homotopy_dynamics import HomotopyEphemerisDynamics
from .synodic_j2000 import SynodicJ2000Transformation

__all__ = [
    "CR3BP_System",
    "LibrationPoint",
    "Dynamics",
    "CR3BP_Dynamics",
    "Orbit",
    "OrbitFamily",
    "propagate_state_at_orbit_time",
    "CoordinateTransformation",
    "SPICEManager",
    "EphemerisSystem",
    "EphemerisDynamics",
    "HomotopyEphemerisDynamics",
    "SynodicJ2000Transformation",
]
