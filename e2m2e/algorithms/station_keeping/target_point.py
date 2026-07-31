"""???? shim?ADR 0011 ????

????? ``e2m2e.algorithm.station_keeping``?????????
"""

from __future__ import annotations

import importlib

_impl = importlib.import_module("e2m2e.algorithm.station_keeping.target_point")


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))
