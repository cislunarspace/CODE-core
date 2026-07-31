"""Lissajous 初猜 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.family``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.family.lissajous_initial_guess import (
    compute_lissajous_initial_guess,
)

__all__ = ["compute_lissajous_initial_guess"]
