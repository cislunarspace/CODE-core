"""星历修正分发模块

通过 ``PatchPointCorrector`` 注册表分发修正方法，
取代原先的 ``if/elif`` 字符串分发。

每种修正方法由一个私有 ``PatchPointCorrector`` 实现包装对应求解器，
将统一参数翻译为求解器特定参数，并将求解器结果重包为
``EphemerisCorrectionResult``。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from e2m2e.mbse.data.enums import BoundaryMode

from .ephemeris_correction_types import (
    EphemerisCorrectionResult,
    PatchPointCorrector,
    UnsupportedCorrectorMethodError,
)
from .multiple_shooting import MultipleShooting
from .two_level_multiple_shooting import TwoLevelMultipleShooting

# ---------------------------------------------------------------------------
# 修正器实现（私有）
# ---------------------------------------------------------------------------


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
            velocity_tolerance=(
                velocity_tolerance if velocity_tolerance is not None else 1e-6
            ),
            boundary=BoundaryMode.FIXED_ENDPOINTS,
            verbose=verbose,
        )
        position_history, velocity_history = _split_residual_history(
            result.residual_history
        )
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


class _HomotopyPatchPointCorrector:
    """包装 ``correct_with_homotopy``，实现 ``PatchPointCorrector`` 接缝。

    同伦修正器的构造参数（``base_bodies``、``lambda_steps`` 等）
    通过构造器注入；``correct`` 只接收统一的求解参数。
    """

    def __init__(
        self,
        dynamics: Any,
        *,
        base_bodies: list[str] | None = None,
        lambda_steps: list[float] | None = None,
        n_workers: int = 1,
        kernel_dir: str | None = None,
        inner_method: str = "standard",
        **_kwargs: Any,
    ) -> None:
        self._dynamics = dynamics
        self._base_bodies = base_bodies
        self._lambda_steps = lambda_steps
        self._n_workers = n_workers
        self._kernel_dir = kernel_dir
        self._inner_method = inner_method

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
        """调用 ``correct_with_homotopy`` 修正 patch points 并返回统一结果。

        构造参数（``base_bodies``、``lambda_steps`` 等）已在构造器注入；
        ``correct`` 仅接收统一的求解参数。
        """
        # 延迟 import 避免循环依赖（homotopy_correction 从本模块导入类型）
        from .homotopy_correction import correct_with_homotopy

        kwargs: dict = {
            "tolerance": tolerance,
            "max_iter": max_iter,
            "n_workers": self._n_workers,
            "kernel_dir": self._kernel_dir,
            "verbose": verbose,
            "inner_method": self._inner_method,
        }
        if self._base_bodies is not None:
            kwargs["base_bodies"] = list(self._base_bodies)
        if self._lambda_steps is not None:
            kwargs["lambda_steps"] = list(self._lambda_steps)
        return correct_with_homotopy(self._dynamics, t_patch, state_patch, **kwargs)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., PatchPointCorrector]] = {
    "standard": lambda dynamics, **kw: _StandardPatchPointCorrector(dynamics, **kw),
    "two_level": lambda dynamics, **kw: _TwoLevelPatchPointCorrector(dynamics, **kw),
    "homotopy": lambda dynamics, **kw: _HomotopyPatchPointCorrector(dynamics, **kw),
}


# ---------------------------------------------------------------------------
# 公开分发函数（保持向后兼容）
# ---------------------------------------------------------------------------


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
    base_bodies: list[str] | None = None,
    lambda_steps: list[float] | None = None,
    inner_method: str = "standard",
) -> EphemerisCorrectionResult:
    """通过注册表分发星历 patch points 修正。

    Args:
        method: 修正方法名称（``"standard"``、``"two_level"``、``"homotopy"``）
        dynamics: 动力学模型对象
        t_patch: 时间节点数组
        state_patch: 状态量数组，形状 ``(N, 6)``
        tolerance: 位置残差收敛容差
        max_iter: 最大迭代次数
        verbose: 是否显示进度条
        n_workers: 并行工作进程/线程数
        kernel_dir: SPICE 内核目录路径
        velocity_tolerance: 速度残差容差（仅 ``"two_level"`` 使用）
        base_bodies: 同伦基础天体列表（仅 ``"homotopy"`` 使用）
        lambda_steps: 同伦 lambda 步长序列（仅 ``"homotopy"`` 使用）
        inner_method: 同伦内部修正方法（仅 ``"homotopy"`` 使用）

    Returns:
        ``EphemerisCorrectionResult``

    Raises:
        UnsupportedCorrectorMethodError: 未知的修正方法名称
    """
    factory = _REGISTRY.get(method)
    if factory is None:
        raise UnsupportedCorrectorMethodError(method, list(_REGISTRY))
    corrector = factory(
        dynamics,
        n_workers=n_workers,
        kernel_dir=kernel_dir,
        base_bodies=base_bodies,
        lambda_steps=lambda_steps,
        inner_method=inner_method,
    )
    return corrector.correct(
        t_patch,
        state_patch,
        max_iter=max_iter,
        tolerance=tolerance,
        velocity_tolerance=velocity_tolerance,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


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
