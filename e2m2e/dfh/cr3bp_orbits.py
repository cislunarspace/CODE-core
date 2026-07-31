"""CR3BP 周期轨道生成 shim（ADR 0011 迁移）。

六类初猜与族行走实现已迁至 ``e2m2e.algorithm.family``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.family import (
    Cr3bpOrbitError,
    design_dro,
    design_halo,
    design_lissajous,
    design_nrho,
    design_triangular,
    earth_moon_system,
)
from e2m2e.data.templates.seed import MOON_RADIUS_KM

__all__ = [
    "Cr3bpOrbitError",
    "design_dro",
    "design_halo",
    "design_nrho",
    "design_lissajous",
    "design_triangular",
    "earth_moon_system",
    "MOON_RADIUS_KM",
]
