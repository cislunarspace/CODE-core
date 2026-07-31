"""3D 对称修正策略 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.family.strategies``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.family.strategies.symmetric_3d import (
    symmetric_3d_fixed_x0,
    symmetric_xz_fixed_x0,
    symmetric_xz_fixed_z0,
)

__all__ = ["symmetric_3d_fixed_x0", "symmetric_xz_fixed_x0", "symmetric_xz_fixed_z0"]
