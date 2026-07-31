"""DFH 摄动开关映射 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.forces.force_mapping``，旧路径保持可用。
"""

from __future__ import annotations

import importlib

_impl = importlib.import_module("e2m2e.algorithm.forces.force_mapping")


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))
