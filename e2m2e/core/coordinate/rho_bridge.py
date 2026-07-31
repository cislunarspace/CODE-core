"""Rho 会合系桥接 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.coordinate``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.coordinate.rho_bridge import (
    _compute_lp_state_j2000,
    _jd_to_et,
    compute_emr_rotation,
    eci_to_rho,
    rho_to_eci,
    tu_to_et,
)

__all__ = [
    "_compute_lp_state_j2000",
    "_jd_to_et",
    "compute_emr_rotation",
    "eci_to_rho",
    "rho_to_eci",
    "tu_to_et",
]
