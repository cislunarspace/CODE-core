"""DFH 功能码 2 轨道保持对齐入口 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.station_keeping.controller``，旧路径保持可用。
"""

from __future__ import annotations

import importlib

_impl = importlib.import_module("e2m2e.algorithm.station_keeping.controller")


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))
