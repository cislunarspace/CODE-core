"""?????? shim?ADR 0011 ????

????? ``e2m2e.algorithm.normal_form``?????????
"""

from __future__ import annotations

import e2m2e.algorithm.normal_form.dynamical_substitution as _impl


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))
