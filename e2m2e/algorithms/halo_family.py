"""Halo 轨道族编排 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.family``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.family.halo_family import (
    generate_halo_family,
    generate_halo_seed_orbit,
    halo_pseudo_arclength_continuation,
)

__all__ = [
    "generate_halo_seed_orbit",
    "generate_halo_family",
    "halo_pseudo_arclength_continuation",
]
