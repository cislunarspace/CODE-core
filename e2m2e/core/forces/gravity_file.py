"""重力场系数文件 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.forces``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.forces.gravity_file import (
    GravityFileData,
    extrapolate_coefficients,
    load_cof_file,
    load_gfc_file,
    load_gravity_file,
)

__all__ = [
    "GravityFileData",
    "load_gfc_file",
    "load_cof_file",
    "load_gravity_file",
    "extrapolate_coefficients",
]
