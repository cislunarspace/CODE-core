"""共享物理常量 shim（ADR 0011 迁移）。

物理常量已迁至 ``e2m2e.data.templates.systems``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.data.templates.systems import AU, KM_TO_M, R_EARTH

__all__ = ["R_EARTH", "AU", "KM_TO_M"]
