"""微分修正策略函数 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.family.strategies``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.family.strategies import (
    CorrectionConfig,
    halo_fixed_x0,
    halo_fixed_z0,
    symmetric_2d_fixed_t,
    symmetric_2d_fixed_x0,
    symmetric_2d_fixed_y0,
    symmetric_3d_fixed_x0,
    symmetric_xz_fixed_x0,
    symmetric_xz_fixed_z0,
)

__all__ = [
    "CorrectionConfig",
    "symmetric_2d_fixed_x0",
    "symmetric_2d_fixed_t",
    "symmetric_2d_fixed_y0",
    "symmetric_3d_fixed_x0",
    "symmetric_xz_fixed_x0",
    "symmetric_xz_fixed_z0",
    "halo_fixed_z0",
    "halo_fixed_x0",
]
