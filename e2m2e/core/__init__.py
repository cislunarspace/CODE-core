"""
e2m2e核心模块

包含三体问题系统、动力学、轨道和坐标变换的核心类。
"""

from . import system
from . import dynamics
from . import orbit
from . import coordinate

from .system import System, CR3BP_System, LibrationPoint
from .dynamics import Dynamics, CR3BP_Dynamics, propagate_state_at_orbit_time
from .orbit import Orbit, OrbitFamily
from .coordinate import CoordinateTransformation

__all__ = [
    "System",
    "CR3BP_System",
    "LibrationPoint",
    "Dynamics",
    "CR3BP_Dynamics",
    "Orbit",
    "OrbitFamily",
    "propagate_state_at_orbit_time",
    "CoordinateTransformation",
]
