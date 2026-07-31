"""????? shim?ADR 0011 ????

????? ``e2m2e.tools.viz``?????????
"""

from __future__ import annotations

import importlib

_impl = importlib.import_module("e2m2e.tools.viz.transfer")


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))
