"""庞加莱截面 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.manifold.sections``，旧路径保持可用。
"""

from __future__ import annotations

import e2m2e.algorithm.manifold.sections as _impl


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))
