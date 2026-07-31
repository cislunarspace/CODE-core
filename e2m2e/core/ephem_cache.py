"""预插值星历缓存 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.data.kernels.ephem_cache``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.data.kernels.ephem_cache import (
    DEFAULT_DT_SECONDS,
    EphemCache,
    build_ephem_cache,
)

__all__ = ["DEFAULT_DT_SECONDS", "EphemCache", "build_ephem_cache"]
