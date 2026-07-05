"""标准多重打靶 patch point 修正器。

包装 ``MultipleShooting``，实现 ``PatchPointCorrector`` 接缝，
将统一参数翻译为求解器特定参数，并将求解器结果重包为
``EphemerisCorrectionResult``。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..ephemeris_correction_types import EphemerisCorrectionResult
from ..multiple_shooting import MultipleShooting


class _StandardPatchPointCorrector:
    """包装 ``MultipleShooting``，实现 ``PatchPointCorrector`` 接缝。"""

    def __init__(
        self,
        dynamics: Any,
        *,
        n_workers: int = 1,
        kernel_dir: str | None = None,
        **_kwargs: Any,
    ) -> None:
        self._solver = MultipleShooting(
            dynamics=dynamics,
            n_workers=n_workers,
            kernel_dir=kernel_dir,
        )

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
        """调用 ``MultipleShooting`` 修正 patch points 并返回统一结果。"""
        result = self._solver.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            var_time=True,
            max_iter=max_iter,
            tolerance=tolerance,
            verbose=verbose,
        )
        return EphemerisCorrectionResult(
            converged=result.converged,
            iterations=result.outer_iterations,
            max_residual=float(result.max_residual),
            residual_history=[float(v) for v in result.residual_history],
            t_patch=result.t_patch,
            state_patch=result.state_patch,
        )
