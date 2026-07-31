"""IAU 2006 岁差章动 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.coordinate``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.coordinate.iau_2006 import (
    iau2000eq_matrix,
    iau2000eq_true_matrix,
    nutation_angles,
    nutation_matrix,
    precession_angles,
    precession_matrix,
)

__all__ = [
    "iau2000eq_matrix",
    "iau2000eq_true_matrix",
    "nutation_angles",
    "nutation_matrix",
    "precession_angles",
    "precession_matrix",
]
