"""两层多重打靶 patch point 修正器。

包装 ``TwoLevelMultipleShooting``，实现 ``PatchPointCorrector`` 接缝，
保留两层修正特有的位置/速度残差诊断字段。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from e2m2e.core.enums import BoundaryMode

from ..ephemeris_correction_types import EphemerisCorrectionResult
from ..two_level_multiple_shooting import TwoLevelMultipleShooting


class _TwoLevelPatchPointCorrector:
    """包装 ``TwoLevelMultipleShooting``，实现 ``PatchPointCorrector`` 接缝。"""

    def __init__(self, dynamics: Any, **_kwargs: Any) -> None:
        self._solver = TwoLevelMultipleShooting(dynamics)

    def correct(
        self,
        t_patch: np.ndarray,
        state_patch: np.ndarray,
        *,
        max_iter: int,
        tolerance: float,
        velocity_tolerance: float | None,
        verbose: bool,
    ) -> EphemerisCorrectionResult:
        """调用 ``TwoLevelMultipleShooting`` 修正 patch points 并返回统一结果。"""
        result = self._solver.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            max_outer_iterations=max_iter,
            position_tolerance=tolerance,
            velocity_tolerance=(velocity_tolerance if velocity_tolerance is not None else 1e-6),
            boundary=BoundaryMode.FIXED_ENDPOINTS,
            verbose=verbose,
        )
        position_history, velocity_history = _split_residual_history(result.residual_history)
        return EphemerisCorrectionResult(
            converged=result.converged,
            iterations=result.outer_iterations,
            max_residual=float(result.final_position_residual),
            residual_history=position_history,
            t_patch=result.t_patch,
            state_patch=result.state_patch,
            velocity_residual=float(result.final_velocity_residual),
            velocity_residual_history=velocity_history,
        )


def _split_residual_history(
    residual_history: Sequence[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """将两层修正的 ``(位置残差, 速度残差)`` 历史拆分为两个列表。"""
    position_history = []
    velocity_history = []
    for position, velocity in residual_history:
        position_history.append(float(position))
        velocity_history.append(float(velocity))
    return position_history, velocity_history
