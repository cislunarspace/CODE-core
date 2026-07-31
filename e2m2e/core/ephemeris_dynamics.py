"""星历动力学模型 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.dynamics``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.dynamics import EphemerisDynamics

__all__ = ["EphemerisDynamics"]
