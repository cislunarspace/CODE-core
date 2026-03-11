"""
e2m2e核心模块

包含三体问题系统、动力学、轨道和坐标变换的核心类。
"""

from . import system
from . import dynamics
from . import orbit
from . import coordinate

from .system import CR3BP_System, LibrationPoint
from .dynamics import CR3BP_Dynamics
from .orbit import Orbit, OrbitFamily
from .coordinate import CoordinateTransformation

__all__ = [
    "CR3BP_System",
    "LibrationPoint",
    "CR3BP_Dynamics",
    "Orbit",
    "OrbitFamily",
    "CoordinateTransformation",
]
