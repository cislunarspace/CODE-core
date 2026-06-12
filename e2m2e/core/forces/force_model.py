"""力模型容器与 Rust 积分器传播实现。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from e2m2e.core.dynamics import Dynamics
from e2m2e.integrators import RkMethod, rk_step

from .exceptions import CoordinateTransformError
from .physical_model import PhysicalModel


class ForceModel(Dynamics):
    """聚合多个 PhysicalModel 并完成传播的动力学容器。

    形式上继承 ``Dynamics``，但 ``propagate()`` 使用 Rust ``rk_step``
    单步步进器实现自适应传播，不依赖 ``scipy.solve_ivp``。
    """

    DEFAULT_MAX_STEP = 60.0  # 秒，用于物理单位传播

    def __init__(
        self,
        system: Any,
        forces: list[PhysicalModel] | None = None,
    ) -> None:
        """初始化 ForceModel。

        Args:
            system: 动力学系统，必须提供 ``coordinate_system``。
            forces: 初始力模型列表，默认空列表。
        """
        super().__init__(system)
        if getattr(system, "coordinate_system", None) is None:
            raise ValueError(
                "ForceModel requires system.coordinate_system to be set."
            )
        self._forces: tuple[PhysicalModel, ...] = tuple()
        if forces is not None:
            for force in forces:
                self.add_force(force)

    @property
    def forces(self) -> tuple[PhysicalModel, ...]:
        """当前聚合的力模型，只读。"""
        return self._forces

    def add_force(self, force: PhysicalModel) -> None:
        """添加一个力模型。"""
        if not isinstance(force, PhysicalModel):
            raise TypeError(
                f"force must be a PhysicalModel, got {type(force).__name__}"
            )
        self._forces = self._forces + (force,)

    def remove_force(self, index: int | PhysicalModel) -> None:
        """移除一个力模型（按索引或对象 identity）。"""
        forces = list(self._forces)
        if isinstance(index, int):
            del forces[index]
        else:
            try:
                forces.remove(index)
            except ValueError as exc:
                raise ValueError("force not found in ForceModel") from exc
        self._forces = tuple(forces)

    def _compute_total_acceleration(
        self,
        t: float,
        state: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """计算所有力模型在当前状态下的总加速度。"""
        total = np.zeros(3, dtype=float)
        for force in self._forces:
            total = total + force.compute_acceleration(t, state, self.system)
        return total

    def equations_of_motion(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """运动方程（兼容 Dynamics 接口）。"""
        return self._eom_func(t, state)

    def _get_eom_func(self, with_stm: bool) -> Callable:
        """返回运动方程函数（兼容 Dynamics 接口）。"""
        if with_stm:
            raise NotImplementedError(
                "ForceModel does not support state transition matrices."
            )
        return self._eom_func

    def _eom_func(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """运动方程闭包。"""
        acceleration = self._compute_total_acceleration(t, state)
        return np.concatenate([state[3:6], acceleration])

    def propagate(
        self,
        initial_state: npt.ArrayLike,
        t_span: tuple[float, float],
        t_eval: npt.ArrayLike | None = None,
        with_stm: bool = False,
        with_jacobi: bool = False,
        *,
        initial_step: float | None = None,
        events: list[Callable[[float, npt.NDArray[np.floating]], float]] | None = None,
        max_steps: int = 100_000,
    ) -> dict[str, Any]:
        """使用 Rust rk_step 传播轨迹。

        Args:
            initial_state: 初始状态向量，形状 (6,)。
            t_span: 时间区间 [t0, tf]，单位为 SPICE et 秒。
            t_eval: 评估时间点数组，默认 linspace(t0, tf, 100)。
            with_stm: 不支持，传 True 抛 NotImplementedError。
            with_jacobi: 不支持，传 True 抛 NotImplementedError。
            initial_step: 初始步长，默认从初始状态估算。
            events: 简单终止事件列表，每个事件返回标量，符号变化时停止。
            max_steps: 最大积分步数，默认 100_000。

        Returns:
            包含 ``time``、``states`` 和 ``terminal_event_index`` 的字典。
        """
        self._raise_for_unsupported(with_stm, with_jacobi)

        if len(t_span) != 2:
            raise ValueError("t_span must be a tuple of (t0, tf)")

        t0, tf = float(t_span[0]), float(t_span[1])
        if tf < t0:
            raise NotImplementedError(
                "ForceModel propagation only supports forward integration (tf >= t0)."
            )
        if tf == t0:
            y0 = np.asarray(initial_state, dtype=float)
            self.last_trajectory = (np.array([t0]), y0.reshape(1, -1))
            return {
                "time": np.array([t0]),
                "states": y0.reshape(1, -1),
                "terminal_event_index": None,
            }

        y = np.asarray(initial_state, dtype=float)
        if y.shape != (self.STATE_DIM,):
            raise ValueError(f"initial_state must have shape ({self.STATE_DIM},)")

        t_eval = self._prepare_t_eval(t0, tf, t_eval)
        max_step = float(self.max_step)
        min_step = 1e-12 * abs(tf - t0)
        tol = float(self.rtol)

        if initial_step is not None:
            if initial_step <= 0:
                raise ValueError("initial_step must be positive")
            h = float(initial_step)
        else:
            h = self._estimate_initial_step(y, t0, tf)

        eom = self._eom_func
        event_funcs = list(events) if events is not None else []
        event_values_prev = [func(t0, y) for func in event_funcs]

        times: list[float] = [t0]
        states: list[npt.NDArray[np.floating]] = [y.copy()]
        terminal_event_index: int | None = None

        t = t0
        eval_index = 1  # t_eval[0] == t0 already recorded
        step_count = 0

        while t < tf:
            step_count += 1
            if step_count > max_steps:
                raise RuntimeError(
                    f"ForceModel propagation exceeded maximum steps ({max_steps})."
                )

            h = min(h, max_step)
            if eval_index < len(t_eval):
                h = min(h, t_eval[eval_index] - t)
            h = max(h, min_step)

            result = rk_step(RkMethod.PD45, t, y, h, tol, eom)

            if result.error <= tol:
                # Accept step
                t_new = t + h
                y_new = np.asarray(result.y_new, dtype=float)

                # Event detection
                for idx, func in enumerate(event_funcs):
                    g_prev = event_values_prev[idx]
                    g_curr = func(t_new, y_new)
                    if g_prev * g_curr < 0:
                        terminal_event_index = idx
                        break
                    event_values_prev[idx] = g_curr

                if terminal_event_index is not None:
                    times.append(t_new)
                    states.append(y_new)
                    break

                t = t_new
                y = y_new

                # Record t_eval points
                while eval_index < len(t_eval) and abs(t - t_eval[eval_index]) < 1e-14:
                    times.append(t)
                    states.append(y.copy())
                    eval_index += 1

                if t >= tf:
                    break

                h = result.h_next
            else:
                # Reject step
                if result.h_next < min_step:
                    raise RuntimeError(
                        "Step size below minimum; integration failed."
                    )
                h = result.h_next

        time_array = np.asarray(times, dtype=float)
        state_array = np.asarray(states, dtype=float)
        self.last_trajectory = (time_array, state_array)

        return {
            "time": time_array,
            "states": state_array,
            "terminal_event_index": terminal_event_index,
        }

    def _prepare_t_eval(
        self, t0: float, tf: float, t_eval: npt.ArrayLike | None
    ) -> npt.NDArray[np.floating]:
        """准备并校验 t_eval 数组。"""
        if t_eval is None:
            return np.linspace(t0, tf, 100)

        t_eval = np.asarray(t_eval, dtype=float)
        if t_eval.size == 0:
            return np.linspace(t0, tf, 100)

        if t_eval.ndim != 1:
            raise ValueError("t_eval must be one-dimensional")

        if np.any(t_eval < t0 - 1e-14) or np.any(t_eval > tf + 1e-14):
            raise ValueError("t_eval must be within t_span")

        if not np.all(np.diff(t_eval) >= -1e-14):
            raise ValueError("t_eval must be monotonically increasing")

        # Append tf if not present, then unique
        combined = np.concatenate([t_eval, [tf]])
        t_eval = np.unique(np.round(combined / 1e-14) * 1e-14)
        # Ensure monotonic and within bounds after rounding
        t_eval = np.clip(t_eval, t0, tf)
        return t_eval

    def _estimate_initial_step(
        self, y: npt.NDArray[np.floating], t0: float, tf: float
    ) -> float:
        """从初始状态估算初始步长。"""
        r = np.linalg.norm(y[:3])
        v = np.linalg.norm(y[3:])
        if r == 0 or v == 0:
            return 1e-6 * abs(tf - t0)
        # Rough orbital period estimate for central motion: 2*pi*r/v
        period = 2.0 * np.pi * r / v
        return period / 100.0

    def _raise_for_unsupported(
        self, with_stm: bool, with_jacobi: bool
    ) -> None:
        if with_stm:
            raise NotImplementedError(
                "ForceModel does not support state transition matrices in this slice."
            )
        if with_jacobi:
            raise NotImplementedError(
                "ForceModel does not support Jacobi constant computation."
            )
