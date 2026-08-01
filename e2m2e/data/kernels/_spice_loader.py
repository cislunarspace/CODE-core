"""SPICE 模块延迟加载工具（数据层内部）。

将 spiceypy 的导入集中管理，避免多个数据层模块各自重复实现延迟加载逻辑，
同时提供线程安全的首次导入保护。源：``core/_spice_loader.py``（ADR 0011
迁移，数据层自足，不依赖旧包）。
"""

from __future__ import annotations

import threading
from types import ModuleType

_spiceypy: ModuleType | None = None
_spiceypy_lock = threading.Lock()


def get_spiceypy() -> ModuleType:
    """延迟加载 spiceypy，仅在首次需要 SPICE 功能时导入。"""
    global _spiceypy
    if _spiceypy is None:
        with _spiceypy_lock:
            if _spiceypy is None:
                import spiceypy as _spiceypy_module

                _spiceypy = _spiceypy_module
    return _spiceypy
