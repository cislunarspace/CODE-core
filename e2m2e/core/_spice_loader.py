"""SPICE 延迟加载工具 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.data.kernels._spice_loader``，旧路径保持可用。
"""

from ..data.kernels._spice_loader import get_spiceypy

__all__ = ["get_spiceypy"]
