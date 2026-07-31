"""GMAT 时间转换 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.coordinate``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.coordinate.gmat_time import (
    J2000_MJD,
    TimeSystemConverter,
)

__all__ = ["TimeSystemConverter", "J2000_MJD"]
