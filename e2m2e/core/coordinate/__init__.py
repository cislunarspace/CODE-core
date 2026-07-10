"""坐标系族：轴、原点、坐标系与各种具体实现。

本子包把坐标系相关的抽象基类与标准实现从 ``e2m2e.core`` 根层收入子目录，
与测试侧 ``tests/core/coordinate/`` 的分层对齐。

对外 API 通过 ``e2m2e.core`` 顶层重导出保持不变（``from e2m2e.core import Axes``
仍可用）；旧路径 ``from e2m2e.core.axes import Axes`` 经 ``e2m2e.core.__init__``
的模块别名兼容。
"""

from __future__ import annotations

from .axes import Axes
from .coordinate_system import CoordinateSystem
from .dynamic_axes import DynamicAxes
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
    "ErfaXysProvider",
    "XysProvider",
    "iau2000eq_matrix",
    "precession_angles",
    "rho_to_eci",
    "eci_to_rho",
]
