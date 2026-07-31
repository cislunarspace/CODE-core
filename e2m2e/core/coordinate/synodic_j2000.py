"""Synodic ↔ J2000 转换器 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.coordinate``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.coordinate import SynodicJ2000System

__all__ = ["SynodicJ2000System"]
