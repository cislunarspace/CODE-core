"""微分修正策略函数。

每个策略函数返回一个不可变的 CorrectionConfig，完整描述修正配置
（对称性、自由变量、约束等）。DifferentialCorrection 类委托给这些函数，
使配置逻辑与迭代求解器分离。
"""

from .axial import axial_fixed_vz0
from .base import CorrectionConfig
from .halo import halo_fixed_x0, halo_fixed_z0
from .lpo import lpo_fixed_x0
from .spo import spo_fixed_x0
from .symmetric_2d import (
    symmetric_2d_fixed_t,
    symmetric_2d_fixed_x0,
    symmetric_2d_fixed_y0,
)
from .symmetric_3d import (
    symmetric_3d_fixed_x0,
    symmetric_xz_fixed_x0,
    symmetric_xz_fixed_z0,
)

__all__ = [
    "axial_fixed_vz0",
    "CorrectionConfig",
    "lpo_fixed_x0",
    "spo_fixed_x0",
    "symmetric_2d_fixed_x0",
    "symmetric_2d_fixed_t",
    "symmetric_2d_fixed_y0",
    "symmetric_3d_fixed_x0",
    "symmetric_xz_fixed_x0",
    "symmetric_xz_fixed_z0",
    "halo_fixed_z0",
    "halo_fixed_x0",
]
