"""二维不变环面构造（拟周期参考轨道内核，#842）。

频闪映射口径的不变闭曲线构造与定义级诊断：入口收周期轨道与目标振幅，
输出相位网格状态、旋转数与第二周期。术语见 CONTEXT.md 的拟周期不变
环面一节，决策记录见 ADR 0063。
"""

from __future__ import annotations

from e2m2e.algorithm.torus.construction import (
    CenterMultiplier,
    InvariantTorus,
    TorusError,
    central_multiplier_pair,
    construct_invariant_torus,
    evaluate_torus_curve,
)
from e2m2e.algorithm.torus.diagnostics import (
    conjugacy_residual,
    invariance_residual,
    torus_long_time_distance,
)

__all__ = [
    "CenterMultiplier",
    "InvariantTorus",
    "TorusError",
    "central_multiplier_pair",
    "conjugacy_residual",
    "construct_invariant_torus",
    "evaluate_torus_curve",
    "invariance_residual",
    "torus_long_time_distance",
]
