"""SPICE 模块延迟加载工具。

将 spiceypy 的导入集中管理，避免多个核心模块各自重复实现延迟加载逻辑，
同时提供线程安全的首次导入保护。
"""

from __future__ import annotations

import threading
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spiceypy


_spiceypy: ModuleType | None = None
_spiceypy_lock = threading.Lock()


def get_spiceypy() -> ModuleType:
    """延迟加载 spiceypy，仅在首次需要 SPICE 功能时导入。

    使用双检锁保证多线程环境下只导入一次，与 SPICEManager 的
    闰秒内核加载锁保持一致的设计理念。
    """
    global _spiceypy
    if _spiceypy is None:
        with _spiceypy_lock:
            if _spiceypy is None:
                import spiceypy as _spiceypy_module

                _spiceypy = _spiceypy_module
    return _spiceypy
