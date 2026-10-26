"""间接低推力单打靶与可替换末端坐标。

动力学采用六维机械状态 ``(r, v)``，自然加速度及其雅可比由调用方注入。
状态、协态与变分传播由 Rust 数值核执行，Python 负责目标映射、阻尼最小二乘、
回溯线搜索与结果解释。固定质量、固定时间和 sigmoid 平滑燃料目标对应
Parrish (2018) 的间接单打靶设置。加速度上限不随质量变化。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ...status import ConvergenceState, FailureCause, ResultStatus

FloatArray = npt.NDArray[np.float64]
NaturalDynamics = Callable[[float, FloatArray], tuple[FloatArray, FloatArray, FloatArray]]
StateMap = Callable[[FloatArray], FloatArray]
AngleMetric = Callable[[FloatArray, FloatArray], float]


class CoordinateTarget(Protocol):
    """将物理末态映射到末端坐标，并标明循环角分量。"""

    target: FloatArray

    def coordinates(self, state: FloatArray) -> FloatArray:
        """返回与物理状态等价的六维末端坐标。"""
        ...

    def angles(self) -> tuple[int, ...]:
        """返回循环角分量的下标。"""
        ...

    def angle_scale(self, state: FloatArray) -> float:
        """返回角误差转换到局部坐标长度的尺度。"""
        ...


@dataclass(frozen=True)
class CartesianTarget:
    """无量纲笛卡尔目标。速度残差可乘统一时间尺度。"""

    target: FloatArray
    velocity_scale: float = 1.0

    def __post_init__(self) -> None:
        if np.asarray(self.target).shape != (6,):
            raise ValueError("target 必须是六维状态")
        if self.velocity_scale <= 0.0:
            raise ValueError("velocity_scale 必须为正数")

    def coordinates(self, state: FloatArray) -> FloatArray:
        result = np.asarray(state, dtype=float).copy()
        result[3:6] *= self.velocity_scale
        return result

    def angles(self) -> tuple[int, ...]:
        return ()

    def angle_scale(self, state: FloatArray) -> float:
        del state
        return 1.0


@dataclass(frozen=True)
class TransformedTarget:
    """由调用方映射构造的局部坐标目标。

    ``state_map`` 将物理末态映射为六维坐标。角度下标显式给出，角误差使用
    最短分支。``angle_metric`` 把角误差转换为局部坐标长度。
    """

    target: FloatArray
    state_map: StateMap
    angle_indices: tuple[int, ...] = ()
    angle_metric: AngleMetric | None = None

    def __post_init__(self) -> None:
        if np.asarray(self.target).shape != (6,):
            raise ValueError("target 必须是六维坐标")

    def coordinates(self, state: FloatArray) -> FloatArray:
        mapped = np.asarray(self.state_map(np.asarray(state, dtype=float)), dtype=float)
        if mapped.shape != (6,) or not np.all(np.isfinite(mapped)):
            raise ValueError("state_map 必须返回有限的六维坐标")
        return mapped

    def angles(self) -> tuple[int, ...]:
        return self.angle_indices

    def angle_scale(self, state: FloatArray) -> float:
        if self.angle_metric is None:
            return 1.0
        value = float(self.angle_metric(np.asarray(state, dtype=float), self.target))
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError("angle_metric 必须返回有限正数")
        return value


@dataclass(frozen=True)
class IndirectShootingConfig:
    """间接传播与求解参数。输入输出单位必须与自然动力学一致。"""

    acceleration_max: float
    epsilon: float = 0.1
    rtol: float = 1e-9
    atol: float = 1e-9
    max_step: float = 0.05
    max_steps: int = 30_000
    max_iterations: int = 80
    residual_tolerance: float = 1e-9
    initial_damping: float = 1e-3

    def __post_init__(self) -> None:
        positive = (
            self.acceleration_max,
            self.epsilon,
            self.rtol,
            self.atol,
            self.max_step,
            self.residual_tolerance,
            self.initial_damping,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("加速度、平滑参数、容差、步长和阻尼必须是有限正数")
        if self.max_steps <= 0 or self.max_iterations <= 0:
            raise ValueError("积分步数与迭代上限必须为正整数")


@dataclass
class IndirectShootingSolution:
    """间接打靶结果。失败时也保留最后可得的物理末态与诊断。"""

    initial_costate_guess: FloatArray
    initial_costate_solution: FloatArray
    final_costate: FloatArray
    final_state: FloatArray
    physical_error: FloatArray
    representation: str
    iterations: int
    evaluations: int
    elapsed_s: float
    residual_history: list[float]
    status: ConvergenceState
    cause: FailureCause
    message: str

    @property
    def result_status(self) -> ResultStatus:
        return ResultStatus(self.status, self.cause, self.message)

    @property
    def converged(self) -> bool:
        return self.status is ConvergenceState.CONVERGED


def terminal_residual(
    final_state: FloatArray,
    target: CoordinateTarget,
) -> FloatArray:
    """用同一坐标映射计算末端残差，角差取最短分支。"""
    mapped = target.coordinates(final_state)
    target_coordinates = (
        target.coordinates(target.target) if isinstance(target, CartesianTarget) else target.target
    )
    target_coordinates = np.asarray(target_coordinates, dtype=float)
    error = mapped - target_coordinates
    for index in target.angles():
        error[index] = (mapped[index] - target_coordinates[index] + np.pi) % (2.0 * np.pi) - np.pi
        error[index] *= target.angle_scale(target_coordinates)
    return error


def _terminal_jacobian(
    final_state: FloatArray,
    target: CoordinateTarget,
) -> FloatArray:
    """中心差分求末端坐标对物理状态的导数，不重复传播轨迹。"""
    if isinstance(target, CartesianTarget):
        return np.diag([1.0, 1.0, 1.0] + [target.velocity_scale] * 3)
    derivative = np.empty((6, 6))
    for column in range(6):
        step = 1e-7 * max(1.0, abs(final_state[column]))
        plus = final_state.copy()
        minus = final_state.copy()
        plus[column] += step
        minus[column] -= step
        derivative[:, column] = (
            terminal_residual(plus, target) - terminal_residual(minus, target)
        ) / (2.0 * step)
    return derivative


def solve_indirect_shooting(
    t0: float,
    tf: float,
    initial_state: FloatArray,
    initial_costate: FloatArray,
    target: CoordinateTarget,
    config: IndirectShootingConfig,
    natural: NaturalDynamics | None = None,
    *,
    cr3bp_mu: float | None = None,
    physical_target: FloatArray | None = None,
    physical_tolerance: FloatArray | None = None,
) -> IndirectShootingSolution:
    """求初始协态，使平滑最优轨迹满足指定末端坐标。

    必须且只能指定 ``natural`` 或 ``cr3bp_mu``。自然回调返回
    ``(a, da_dr, da_dv)``，其中加速度为三维，两个雅可比为 ``(3, 3)``。
    ``physical_target`` 和六维 ``physical_tolerance`` 可独立于坐标残差定义
    物理验收。未提供时以 ``residual_tolerance`` 判定收敛。
    """
    import time

    from ... import integrators

    started = time.perf_counter()
    x0 = np.asarray(initial_state, dtype=float)
    lambda0 = np.asarray(initial_costate, dtype=float)
    if x0.shape != (6,) or lambda0.shape != (6,):
        raise ValueError("initial_state 与 initial_costate 必须为六维")
    if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(lambda0)):
        raise ValueError("初始状态和协态必须有限")
    if (natural is None) == (cr3bp_mu is None):
        raise ValueError("必须且只能指定 natural 或 cr3bp_mu")
    if not np.isfinite(tf) or tf <= t0:
        raise ValueError("tf 必须大于 t0")
    if (physical_target is None) != (physical_tolerance is None):
        raise ValueError("physical_target 与 physical_tolerance 必须同时提供")
    evaluations = 0

    def evaluate(guess: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        nonlocal evaluations
        evaluations += 1
        if cr3bp_mu is not None:
            result = integrators.propagate_cr3bp_indirect_py(
                cr3bp_mu,
                t0,
                tf,
                x0,
                guess,
                config.acceleration_max,
                config.epsilon,
                rtol=config.rtol,
                atol=config.atol,
                max_step=config.max_step,
                max_steps=config.max_steps,
                with_jacobian=True,
            )
        else:
            if natural is None:
                raise ValueError("自然动力学回调不能为空")

            def callback(
                time: float, state: list[float]
            ) -> tuple[list[float], list[float], list[float]]:
                acceleration, ar, av = natural(time, np.asarray(state, dtype=float))
                return (
                    np.asarray(acceleration, dtype=float).tolist(),
                    np.asarray(ar, dtype=float).ravel().tolist(),
                    np.asarray(av, dtype=float).ravel().tolist(),
                )

            result = integrators.propagate_indirect_py(
                t0,
                tf,
                x0,
                guess,
                config.acceleration_max,
                config.epsilon,
                callback,
                rtol=config.rtol,
                atol=config.atol,
                max_step=config.max_step,
                max_steps=config.max_steps,
                with_jacobian=True,
            )
        state = np.asarray(result["state"], dtype=float)
        phi = np.asarray(result["jacobian"], dtype=float)
        residual = terminal_residual(state, target)
        derivative = _terminal_jacobian(state, target) @ phi
        return residual, derivative, state, np.asarray(result["costate"], dtype=float)

    guess = lambda0.copy()
    state = x0.copy()
    final_costate = np.full(6, np.nan)
    residual = np.full(6, np.inf)
    history: list[float] = []
    damping = config.initial_damping
    status = ConvergenceState.MAX_ITERATIONS
    cause = FailureCause.MAX_ITERATIONS_REACHED
    message = "达到迭代上限"
    iterations = 0
    current: tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None = None
    try:
        current = evaluate(guess)
        residual, _, state, _ = current
        history.append(float(np.linalg.norm(residual)))
        for iteration in range(config.max_iterations):
            iterations = iteration + 1
            converged: bool
            if physical_target is None:
                converged = bool(np.linalg.norm(residual) <= config.residual_tolerance)
                physical_error = residual.copy()
            else:
                physical_error = state - np.asarray(physical_target, dtype=float)
                converged = bool(
                    np.all(np.abs(physical_error) <= np.asarray(physical_tolerance, dtype=float))
                )
            if converged:
                status = ConvergenceState.CONVERGED
                final_costate = current[3]
                cause = FailureCause.NONE
                message = "末端误差满足容差"
                break
            residual, derivative, state, _ = current
            gram = derivative @ derivative.T
            regularization = damping * max(float(np.mean(np.diag(gram))), 1e-30)
            try:
                step = -derivative.T @ np.linalg.solve(gram + regularization * np.eye(6), residual)
            except np.linalg.LinAlgError:
                status = ConvergenceState.FAILED
                cause = FailureCause.SINGULAR_JACOBIAN
                message = "末端雅可比无法求解"
                break
            accepted = False
            alpha = 1.0
            norm = float(np.linalg.norm(residual))
            for _ in range(12):
                candidate = evaluate(guess + alpha * step)
                if np.linalg.norm(candidate[0]) < norm:
                    guess = guess + alpha * step
                    current = candidate
                    residual, _, state, _ = candidate
                    history.append(float(np.linalg.norm(residual)))
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                damping = max(damping * 0.3, 1e-12)
            else:
                damping *= 10.0
                if damping > 1e12:
                    status = ConvergenceState.STAGNATED
                    cause = FailureCause.STAGNATION_DETECTED
                    message = "回溯线搜索未找到下降步"
                    break
    except RuntimeError as error:
        status = ConvergenceState.FAILED
        cause = FailureCause.INTEGRATION_FAILED
        message = f"传播失败：{error}"
    except ValueError as error:
        status = ConvergenceState.FAILED
        cause = FailureCause.INVALID_INPUT
        message = f"末端坐标或输入参数无效：{error}"
    elapsed = time.perf_counter() - started
    if status is ConvergenceState.MAX_ITERATIONS and physical_target is None:
        converged = bool(np.all(np.isfinite(residual))) and bool(
            np.linalg.norm(residual) <= config.residual_tolerance
        )
        physical_error = residual.copy()
    elif status is ConvergenceState.MAX_ITERATIONS:
        physical_error = state - np.asarray(physical_target, dtype=float)
        converged = bool(np.all(np.isfinite(physical_error))) and bool(
            np.all(np.abs(physical_error) <= np.asarray(physical_tolerance, dtype=float))
        )
    else:
        converged = False
        physical_error = (
            state - np.asarray(physical_target, dtype=float)
            if physical_target is not None
            else residual.copy()
        )
    if converged:
        status = ConvergenceState.CONVERGED
        final_costate = current[3] if current is not None else final_costate
        cause = FailureCause.NONE
        message = "末端误差满足容差"
    return IndirectShootingSolution(
        initial_costate_guess=lambda0,
        initial_costate_solution=guess,
        final_costate=final_costate,
        final_state=state,
        physical_error=physical_error,
        representation=type(target).__name__,
        iterations=iterations,
        evaluations=evaluations,
        elapsed_s=elapsed,
        residual_history=history,
        status=status,
        cause=cause,
        message=message,
    )


__all__ = [
    "CartesianTarget",
    "CoordinateTarget",
    "IndirectShootingConfig",
    "IndirectShootingSolution",
    "TransformedTarget",
    "solve_indirect_shooting",
    "terminal_residual",
]
