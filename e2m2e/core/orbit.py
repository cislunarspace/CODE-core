"""轨道数据容器 shim（ADR 0011 迁移）。

``Orbit``/``OrbitFamily`` 完整实现已迁至 ``e2m2e.data.types.orbit``，
旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.data.types.orbit import Orbit, OrbitFamily

__all__ = ["Orbit", "OrbitFamily"]
