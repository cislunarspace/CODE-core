"""固体潮力模型 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.forces``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.forces.earth_tide import (
    _K_EARTH,
    _K_PLUS_EARTH,
    load_love_number_file,
    permanent_tide_correction,
    pole_tide,
    solid_tide_step1,
    solid_tide_step2,
)

__all__ = [
    "_K_EARTH",
    "_K_PLUS_EARTH",
    "load_love_number_file",
    "permanent_tide_correction",
    "pole_tide",
    "solid_tide_step1",
    "solid_tide_step2",
]
