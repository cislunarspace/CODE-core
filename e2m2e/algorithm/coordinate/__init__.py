"""坐标转换算法：IAU2006/synodic↔J2000/GCRS↔EBCRS。

转换**算法**归这里（ADR 0011 迁移，源：``core/coordinate/``），
``data/frames/`` 只留数据（EOP/闰秒/历表句柄，ADR 0015）。强化现有
Axes/Origin/CoordinateSystem 抽象（不新增 Frame 抽象）：所有坐标系表达
为 Axes + Origin + CoordinateSystem，时空间联合转换作为 CoordinateSystem
扩展方法。转换算法最终留 Python，Rust 下沉是后续性能优化。
"""

from __future__ import annotations

from .axes import Axes
from .coordinate_system import CoordinateSystem
from .dynamic_axes import DynamicAxes
from .gcrs_ebcrs import GCRSEBCRSSystem
from .gmat_itrf import GmatItrfReduction
from .gmat_time import TimeSystemConverter
from .iau_2006 import iau2000eq_matrix, precession_angles
from .origin import Origin
from .rho_bridge import eci_to_rho, rho_to_eci
from .standard_axes import (
    GMATITRFAxes,
    IAU2000EqAxes,
    ICRSAxes,
    ITRFApproxAxes,
    ITRFAxes,
    ITRFSpiceAxes,
    standard_icrf,
    standard_itrf,
)
from .standard_dynamic_axes import LVLHAxes, VNBAxes
from .standard_origins import CelestialBodyOrigin, InertialOrigin
from .synodic_axes import SynodicAxes
from .synodic_j2000 import SynodicJ2000System
from .xys import ErfaXysProvider, XysProvider

__all__ = [
    "Axes",
    "Origin",
    "CoordinateSystem",
    "DynamicAxes",
    "ICRSAxes",
    "IAU2000EqAxes",
    "ITRFSpiceAxes",
    "ITRFAxes",
    "GMATITRFAxes",
    "ITRFApproxAxes",
    "standard_icrf",
    "standard_itrf",
    "LVLHAxes",
    "VNBAxes",
    "CelestialBodyOrigin",
    "InertialOrigin",
    "SynodicAxes",
    "SynodicJ2000System",
    "GCRSEBCRSSystem",
    "GmatItrfReduction",
    "TimeSystemConverter",
    "ErfaXysProvider",
    "XysProvider",
    "iau2000eq_matrix",
    "precession_angles",
    "rho_to_eci",
    "eci_to_rho",
]
