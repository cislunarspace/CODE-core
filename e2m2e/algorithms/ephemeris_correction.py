from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .multiple_shooting import MultipleShooting
from .two_level_multiple_shooting import TwoLevelMultipleShooting


@dataclass(frozen=True)
class EphemerisCorrectionResult:
    converged: bool
    iterations: int
    max_residual: float
    residual_history: list[float]
    t_patch: np.ndarray
    state_patch: np.ndarray
    velocity_residual: float | None = None
    velocity_residual_history: list[float] | None = None


def correct_ephemeris_patch_points(
    method: str,
    dynamics: Any,
    t_patch: np.ndarray,
    state_patch: np.ndarray,
    *,
    tolerance: float,
    max_iter: int,
    verbose: bool,
    n_workers: int,
    kernel_dir: str,
    velocity_tolerance: float | None = None,
) -> EphemerisCorrectionResult:
    if method == "homotopy":
        raise NotImplementedError("homotopy correction is not fully implemented")
    if method == "standard":
        solver = MultipleShooting(
            dynamics=dynamics,
            n_workers=n_workers,
            kernel_dir=kernel_dir,
        )
        result = solver.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            var_time=True,
            max_iter=max_iter,
            tolerance=tolerance,
            verbose=verbose,
        )
        return EphemerisCorrectionResult(
            converged=result.converged,
            iterations=result.iterations,
            max_residual=float(result.max_residual),
            residual_history=[float(value) for value in result.residual_history],
            t_patch=result.t_patch,
            state_patch=result.state_patch,
        )
    if method == "two_level":
        two_level_solver = TwoLevelMultipleShooting(dynamics)
        two_level_result = two_level_solver.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            max_outer_iterations=max_iter,
            position_tolerance=tolerance,
            velocity_tolerance=(velocity_tolerance if velocity_tolerance is not None else 1e-6),
            boundary="fixed_endpoints",
            verbose=verbose,
        )
        position_history, velocity_history = _split_residual_history(
            two_level_result.residual_history
        )
        return EphemerisCorrectionResult(
            converged=two_level_result.converged,
            iterations=two_level_result.outer_iterations,
            max_residual=float(two_level_result.final_position_residual),
            residual_history=position_history,
            t_patch=two_level_result.t_patch,
            state_patch=two_level_result.state_patch,
            velocity_residual=float(two_level_result.final_velocity_residual),
            velocity_residual_history=velocity_history,
        )
    raise ValueError(f"unsupported correction method: {method}")


def _split_residual_history(
    residual_history: Sequence[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    position_history = []
    velocity_history = []
    for position, velocity in residual_history:
        position_history.append(float(position))
        velocity_history.append(float(velocity))
    return position_history, velocity_history
