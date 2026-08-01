"""同伦星历修正模块。

通过固定步长的 lambda 权重序列，将 patch points 轨迹从基础天体集
（如 ``["EARTH", "MOON"]``）逐步过渡到完整天体集
（如 ``["EARTH", "MOON", "SUN"]``）。

每一步调用内部修正器（标准多重打靶或两层多重打靶），
以上一步的 ``(t_patch, state_patch)`` 作为初值猜测，
无论上一步是否收敛。

动力学加权由 ``HomotopyEphemerisDynamics`` 完成，
它在基础集与完整集之间线性插值加速度与雅可比矩阵：

    a_lambda = a_base + lambda * (a_full - a_base)
    J_lambda = J_base + lambda * (J_full - J_base)

插值仅发生在天体分组维度，不涉及坐标变换或 CR3BP 动力学。
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ...data.templates.enums import BoundaryMode
from ..dynamics import EphemerisDynamics, EphemerisSystem
from ..solver.multiple_shooting import MultipleShooting, MultipleShootingResult
from ..solver.two_level_multiple_shooting import (
    TwoLevelMultipleShooting,
    TwoLevelMultipleShootingResult,
)
from .types import EphemerisCorrectionResult

DEFAULT_LAMBDA_STEPS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)


class HomotopyEphemerisDynamics(EphemerisDynamics):
    """带单一 lambda 线性混合权重的星历动力学。

    加速度与雅可比矩阵按 ``a_base + lambda * (a_full - a_base)`` 计算。
    基础集是完整天体列表的子集 ``base_bodies``；完整集即动力学对象的正常天体列表。
    两者共享相同的 SPICE 原点与坐标框架，因此 ``_compute_acc_and_jacobian`` 的调用结构可直接复用。
    """

    def __init__(
        self,
        system: EphemerisSystem,
        base_bodies: list[str],
        lambda_weight: float,
    ) -> None:
        # Lambda must be 0 <= lambda <= 1 by contract; it is set on the
        # full-dynamics object so that the parent class's EoM routine
        # reuses our interpolation.
        if not 0.0 <= lambda_weight <= 1.0:
            raise ValueError(f"lambda_weight must be in [0, 1], got {lambda_weight}")
        super().__init__(system)
        self.base_bodies = list(base_bodies)
        self.lambda_weight = float(lambda_weight)

        # Build the parallel base-dynamics object on the same SPICE/origin/frame
        # but with bodies reduced to `base_bodies`. origin must be a member of
        # base_bodies (validation lives in correct_with_homotopy).
        self.base_dynamics = EphemerisDynamics(
            EphemerisSystem(
                bodies=list(base_bodies),
                spice=system.spice,
                origin=system.origin,
                frame=system.frame,
            )
        )
        # Mirror integration parameters onto the base dynamics so that any
        # propagation that goes through it matches tolerances/steps.
        self.base_dynamics.rtol = self.rtol
        self.base_dynamics.atol = self.atol
        self.base_dynamics.max_step = self.max_step
        self.base_dynamics.integrator = self.integrator

    def _compute_acc_and_jacobian(
        self,
        t: float,
        r_sc: npt.NDArray[np.floating],
        need_jacobian: bool = False,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
        """在基础集与完整集之间线性插值加速度与雅可比矩阵。"""
        acc_base, jac_base = self.base_dynamics._compute_acc_and_jacobian(t, r_sc, need_jacobian)
        acc_full, jac_full = super()._compute_acc_and_jacobian(t, r_sc, need_jacobian)
        lam = self.lambda_weight
        acc = acc_base + lam * (acc_full - acc_base)
        jac: np.ndarray | None = None
        if need_jacobian:
            assert jac_base is not None and jac_full is not None
            jac = jac_base + lam * (jac_full - jac_base)
        return acc, jac


def _validate_base_bodies(dynamics: EphemerisDynamics, base_bodies: list[str]) -> None:
    """校验 base_bodies 是完整天体列表的子集且包含原点。"""
    full_bodies = list(dynamics.system.bodies)
    full_set = set(full_bodies)
    base_set = set(base_bodies)
    if not base_set.issubset(full_set):
        missing = sorted(base_set - full_set)
        raise ValueError(
            f"base_bodies {base_bodies} must be a subset of system.bodies "
            f"{full_bodies}; unknown: {missing}"
        )
    if dynamics.system.origin not in base_set:
        raise ValueError(
            f"base_bodies {base_bodies} must include origin {dynamics.system.origin!r}"
        )


def _validate_lambda_steps(lambda_steps: list[float]) -> None:
    """校验 lambda 步长序列非空、严格递增、在 [0, 1] 范围内且终值为 1.0。"""
    if not lambda_steps:
        raise ValueError("lambda_steps must not be empty")
    for value in lambda_steps:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"lambda_steps values must be in [0, 1], got {value}")
    for i in range(len(lambda_steps) - 1):
        if lambda_steps[i + 1] <= lambda_steps[i]:
            raise ValueError(f"lambda_steps must be strictly increasing, got {lambda_steps}")
    if lambda_steps[-1] != 1.0:
        raise ValueError(
            f"lambda_steps must end at 1.0 (final step is the full dynamics), "
            f"got {lambda_steps[-1]}"
        )


def correct_with_homotopy(
    dynamics: EphemerisDynamics,
    t_patch: np.ndarray,
    state_patch: np.ndarray,
    *,
    tolerance: float,
    max_iter: int,
    n_workers: int,
    kernel_dir: str,
    base_bodies: list[str],
    lambda_steps: list[float] | None = None,
    inner_method: str = "standard",
    velocity_tolerance: float | None = None,
    verbose: bool = False,
) -> EphemerisCorrectionResult:
    """按固定 lambda 步长序列驱动基础集到完整集的过渡修正。

    每一步调用内部修正器（默认 ``MultipleShooting``，
    ``inner_method="two_level"`` 时为 ``TwoLevelMultipleShooting``），
    以上一步输出的 ``(t_patch, state_patch)`` 作为初值猜测。
    中间步使用 ``tolerance * 10`` 的宽松容差；最终 ``lambda=1.0`` 步使用严格容差。

    ``EphemerisCorrectionResult`` 汇总语义：
      - standard：``converged`` 取最后一步结果，``iterations`` 累加，
        ``max_residual`` 取最后一步，``residual_history`` 扁平化。
      - two_level：同上，另含 ``velocity_residual`` 与
        ``velocity_residual_history`` （来自两层历史对）。
    """
    if inner_method == "homotopy":
        raise ValueError("inner_method='homotopy' is not allowed (would recurse)")
    if inner_method not in ("standard", "two_level"):
        raise ValueError(
            f"unsupported inner_method: {inner_method!r}; expected 'standard' or 'two_level'"
        )

    steps = list(lambda_steps) if lambda_steps is not None else list(DEFAULT_LAMBDA_STEPS)
    _validate_base_bodies(dynamics, base_bodies)
    _validate_lambda_steps(steps)

    t_work = np.asarray(t_patch, dtype=float).copy()
    state_work = np.asarray(state_patch, dtype=float).copy()
    position_histories: list[float] = []
    velocity_histories: list[float] = []
    iterations_total = 0
    final_converged = False
    final_max_residual = float("inf")
    final_velocity_residual: float | None = None
    last_t = t_work
    last_state = state_work

    for step_index, lam in enumerate(steps):
        step_tol = tolerance if lam == 1.0 else tolerance * 10.0
        step_dynamics = HomotopyEphemerisDynamics(
            system=dynamics.system,
            base_bodies=base_bodies,
            lambda_weight=lam,
        )
        # Mirror integration parameters from the supplied dynamics.
        step_dynamics.rtol = dynamics.rtol
        step_dynamics.atol = dynamics.atol
        step_dynamics.max_step = dynamics.max_step
        step_dynamics.integrator = dynamics.integrator

        try:
            solver: MultipleShooting | TwoLevelMultipleShooting
            step_result: MultipleShootingResult | TwoLevelMultipleShootingResult
            if inner_method == "standard":
                solver = MultipleShooting(
                    dynamics=step_dynamics,
                    n_workers=n_workers,
                    kernel_dir=kernel_dir,
                )
                step_result = solver.correct(
                    t_patch=t_work,
                    state_patch=state_work,
                    var_time=True,
                    max_iter=max_iter,
                    tolerance=step_tol,
                    verbose=verbose,
                )
                position_histories.extend(float(v) for v in step_result.residual_history)
                iterations_total += int(step_result.outer_iterations)
                final_converged = bool(step_result.converged)
                final_max_residual = float(step_result.max_residual)
                last_t = step_result.t_patch
                last_state = step_result.state_patch
            else:  # two_level
                solver = TwoLevelMultipleShooting(step_dynamics)
                vel_tol = velocity_tolerance if velocity_tolerance is not None else 1e-6
                step_result = solver.correct(
                    t_patch=t_work,
                    state_patch=state_work,
                    max_outer_iterations=max_iter,
                    position_tolerance=step_tol,
                    velocity_tolerance=vel_tol,
                    boundary=BoundaryMode.FIXED_ENDPOINTS,
                    verbose=verbose,
                )
                # 两层求解器的运行时契约：residual_history 为 (位置, 速度) 元组列表，
                # final_position_residual / final_velocity_residual 为末次外层迭代残差。
                # 此处 step_result 在测试中可能被 mock 成 SimpleNamespace，
                # 因此不在运行时做 isinstance 校验，只用 cast 告诉 mypy 期望类型。
                two_level_result = cast(TwoLevelMultipleShootingResult, step_result)
                pos_hist, vel_hist = _split_residual_history(two_level_result.residual_history)
                position_histories.extend(pos_hist)
                velocity_histories.extend(vel_hist)
                iterations_total += int(two_level_result.outer_iterations)
                final_converged = bool(two_level_result.converged)
                final_max_residual = float(two_level_result.final_position_residual)
                final_velocity_residual = float(two_level_result.final_velocity_residual)
                last_t = two_level_result.t_patch
                last_state = two_level_result.state_patch
        except Exception as exc:
            raise RuntimeError(
                f"homotopy lambda step {step_index} (lambda={lam}, "
                f"inner_method={inner_method}) failed: {exc}"
            ) from exc

        # Use the latest result as the next step's initial guess, even if
        # the current step did not converge — the next lambda is closer to
        # 1.0 and may pull the trajectory back into the basin.
        t_work = np.asarray(step_result.t_patch, dtype=float).copy()
        state_work = np.asarray(step_result.state_patch, dtype=float).copy()

    return EphemerisCorrectionResult(
        converged=final_converged,
        iterations=iterations_total,
        max_residual=final_max_residual,
        residual_history=position_histories,
        t_patch=last_t,
        state_patch=last_state,
        velocity_residual=final_velocity_residual,
        velocity_residual_history=velocity_histories if velocity_histories else None,
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


def _split_residual_history(
    residual_history: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """将两层修正的 ``(位置残差, 速度残差)`` 历史拆分为两个列表。"""
    pos: list[float] = []
    vel: list[float] = []
    for pair in residual_history:
        p, v = pair
        pos.append(float(p))
        vel.append(float(v))
    return pos, vel
