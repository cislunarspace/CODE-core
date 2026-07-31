"""动力学系统抽象基类 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.dynamics``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.dynamics import System

__all__ = ["System"]
