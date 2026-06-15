"""力模型容器与 Rust 积分器传播实现。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from e2m2e.core.dynamics import Dynamics
from e2m2e.integrators import RkMethod, rk_step

from .physical_model import PhysicalModel
from .thrust import BurnApplication, ImpulsiveBurn


@dataclass(frozen=True)
class ForceEntry:
    """容器内单个力模型的注册记录。"""

    name: str
    force: PhysicalModel
    enabled: bool = True


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
        self._entries: tuple[ForceEntry, ...] = ()
        if forces is not None:
            for force in forces:
                self.add_force(force)

    @property
    def forces(self) -> tuple[PhysicalModel, ...]:
        """当前聚合的力模型，只读（含已 disable 的项）。"""
        return tuple(entry.force for entry in self._entries)

    def add_force(self, force: PhysicalModel, name: str | None = None) -> None:
        """添加一个力模型。

        Args:
            force: 待添加的力模型。
            name: 力模型的名字。缺省时按类名自动生成，遇同类自动消歧
                （``Foo``、``Foo_2``、``Foo_3``…）。显式给出且与已有名字
                冲突时抛 ``ValueError``。
        """
        if not isinstance(force, PhysicalModel):
            raise TypeError(
                f"force must be a PhysicalModel, got {type(force).__name__}"
            )
        if name is None:
            name = self._auto_name(force)
        elif any(entry.name == name for entry in self._entries):
            raise ValueError(f"force name {name!r} already exists in ForceModel")
        self._entries = self._entries + (ForceEntry(name, force, True),)

    def _auto_name(self, force: PhysicalModel) -> str:
        """按类名生成唯一名字，遇同类自动加序号后缀。"""
        base = type(force).__name__
        if not any(entry.name == base for entry in self._entries):
            return base
        suffix = 2
        while any(entry.name == f"{base}_{suffix}" for entry in self._entries):
            suffix += 1
        return f"{base}_{suffix}"

    def get_force(self, name: str) -> PhysicalModel:
        """按名取力模型；不存在抛 ``KeyError``。"""
        for entry in self._entries:
            if entry.name == name:
                return entry.force
        raise KeyError(name)

    def list_forces(self) -> list[ForceEntry]:
        """返回所有力模型的注册记录（含已 disable 的项）。

        与 ``forces`` 属性的区别：本方法暴露 ``name`` 与 ``enabled`` 两个维度。
        """
        return list(self._entries)

    def enable(self, name: str) -> None:
        """按名启用一个力模型；不存在抛 ``KeyError``。"""
        self._set_enabled(name, True)

    def disable(self, name: str) -> None:
        """按名禁用一个力模型（跳过加速度计算，但保留在容器内）。

        不存在抛 ``KeyError``。
        """
        self._set_enabled(name, False)

    def _set_enabled(self, name: str, enabled: bool) -> None:
        """翻转指定名字力模型的 enabled 标志（不可变替换）。"""
        if not any(entry.name == name for entry in self._entries):
            raise KeyError(name)
        self._entries = tuple(
            ForceEntry(entry.name, entry.force, enabled) if entry.name == name else entry
            for entry in self._entries
        )

    # --- 配置驱动（ADR 0004）---

    _CONFIG_VERSION = 1

    def to_config(self) -> dict[str, Any]:
        """序列化为配置字典 ``{version, forces: [...]}``。

        每条力经 ``force_config.serialize_force`` 转 ``{type, params}``，
        容器补 ``name`` 与 ``enabled``。round-trip 契约见 ADR 0004。
        """
        from .force_config import serialize_force

        forces_config: list[dict[str, Any]] = []
        for entry in self._entries:
            single = serialize_force(entry.force)
            forces_config.append(
                {
                    "name": entry.name,
                    "type": single["type"],
                    "enabled": entry.enabled,
                    "params": single["params"],
                }
            )
        return {"version": self._CONFIG_VERSION, "forces": forces_config}

    @classmethod
    def from_config(
        cls, config: dict[str, Any], system: Any
    ) -> ForceModel:
        """从配置字典构建 ``ForceModel``。

        校验 ``version``，逐条 ``force_config.build_force`` 构造并按 ``name``
        注册；``enabled: false`` 的条目构造后立即 disable。
        """
        from .force_config import build_force

        version = config.get("version")
        if version != cls._CONFIG_VERSION:
            raise ValueError(
                f"unsupported config version {version!r}; "
                f"expected {cls._CONFIG_VERSION}"
            )
        fm = cls(system)
        for entry in config.get("forces", []):
            force = build_force(entry["type"], entry.get("params", {}))
            fm.add_force(force, name=entry["name"])
            if not entry.get("enabled", True):
                fm.disable(entry["name"])
        return fm

    def remove_force(self, index: int | PhysicalModel | str) -> None:
        """移除一个力模型（按索引、实例 identity 或名字）。"""
        entries = list(self._entries)
        if isinstance(index, int):
            del entries[index]
        elif isinstance(index, str):
            for i, entry in enumerate(entries):
                if entry.name == index:
                    del entries[i]
                    break
            else:
                raise ValueError(f"force name {index!r} not found in ForceModel")
        else:
            for i, entry in enumerate(entries):
                if entry.force is index:
                    del entries[i]
                    break
            else:
                raise ValueError("force not found in ForceModel")
        self._entries = tuple(entries)

    def _compute_total_acceleration(
        self,
        t: float,
        state: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """计算所有启用力模型在当前状态下的总加速度。"""
        total = np.zeros(3, dtype=float)
        for entry in self._entries:
            if not entry.enabled:
                continue
            total = total + entry.force.compute_acceleration(t, state, self.system)
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

    def propagate_maneuvers(
        self,
        initial_state: npt.ArrayLike,
        t_span: tuple[float, float],
        burns: list[ImpulsiveBurn],
        *,
        initial_step: float | None = None,
        max_steps: int = 100_000,
    ) -> dict[str, Any]:
        """带脉冲机动的传播：coast 段之间在 burn epoch 处施加 Δv。

        按 epoch 排序 burns，依次 coast → 施加 Δv → 续传。burn epoch 处
        输出行携带 post-burn 速度（丢 pre-burn 行，无重复 epoch）。
        """
        t0, tf = float(t_span[0]), float(t_span[1])
        y = np.asarray(initial_state, dtype=float)

        _eps = 1e-12
        for burn in burns:
            if burn.epoch < t0 - _eps or burn.epoch > tf + _eps:
                raise ValueError(
                    f"burn epoch {burn.epoch} outside t_span ({t0}, {tf})"
                )

        sorted_burns = sorted(burns, key=lambda b: b.epoch)

        segments: list[
            tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        ] = []
        burn_meta: list[
            tuple[
                float,
                npt.NDArray[np.floating],
                npt.NDArray[np.floating],
                npt.NDArray[np.floating],
            ]
        ] = []
        current_t = t0
        for burn in sorted_burns:
            seg = self.propagate(
                y,
                (current_t, burn.epoch),
                initial_step=initial_step,
                max_steps=max_steps,
            )
            segments.append((seg["time"], seg["states"]))
            y = np.asarray(seg["states"][-1], dtype=float).copy()
            current_t = burn.epoch
            velocity_before = y[3:6].copy()
            delta_v = np.asarray(burn.delta_v, dtype=float)
            y[3:6] = y[3:6] + delta_v
            velocity_after = y[3:6].copy()
            burn_meta.append((burn.epoch, delta_v, velocity_before, velocity_after))

        seg = self.propagate(
            y,
            (current_t, tf),
            initial_step=initial_step,
            max_steps=max_steps,
        )
        segments.append((seg["time"], seg["states"]))

        # 拼接：首段全留；后续段替换运行结果的末行（pre-burn）为本段首行（post-burn）
        time = segments[0][0]
        states = segments[0][1]
        applied: list[BurnApplication] = []
        for i, (seg_time, seg_states) in enumerate(segments[1:]):
            post_burn_index = len(states) - 1
            epoch_i, dv_i, vb_i, va_i = burn_meta[i]
            applied.append(
                BurnApplication(
                    index=post_burn_index,
                    epoch=epoch_i,
                    delta_v=dv_i,
                    velocity_before=vb_i,
                    velocity_after=va_i,
                )
            )
            time = np.concatenate([time[:-1], seg_time])
            states = np.concatenate([states[:-1], seg_states], axis=0)

        self.last_trajectory = (time, states)
        return {
            "time": time,
            "states": states,
            "burns": applied,
            "terminal_event_index": None,
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
