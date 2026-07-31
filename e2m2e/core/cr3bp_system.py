"""圆型限制性三体问题系统 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.dynamics``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.dynamics import CR3BP_System, LibrationPoint

__all__ = ["CR3BP_System", "LibrationPoint"]
