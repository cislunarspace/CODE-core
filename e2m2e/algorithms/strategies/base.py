"""修正策略基类 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.family.strategies``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.family.strategies.base import CorrectionConfig

__all__ = ["CorrectionConfig"]
