"""低推力多段直接打靶求解器。

在地基 ``VariableMassFiniteBurn`` / ``propagate_compiled_lowthrust``（7D 可变
质量受控传播，commit ``b66fa88``）之上，建第一个低推力最优控制闭环求解器：
单弧多段直接打靶，min-fuel，固定时间，段内常量控制。

## 数学模型

固定初态 ``[r0, v0, m0]``，把 ``[t0, tf]`` 均分成 N 段。决策变量为各段常量
控制 ``(throttle_i, u_i)``，方向存原始向量（内部归一化），共 ``4N`` 维。传播
为接龙：段 i 在段内常量控制下积分，段末 7D 状态作段 i+1 初态，串行到末态。
目标为最大化末态质量（min-fuel）；约束为末态位置速度匹配目标（6 条等式）。

这是航天界低推力「先打靶后配点」的标准入门路线。详见
``docs/plans/lowthrust-shooting-prd.md``。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, minimize

from ..core.forces import PhysicalModel

if TYPE_CHECKING:
    from ..core.system import System


@dataclass(frozen=True)
class EngineConfig:
    """推进配置：最大推力与比冲。

    与 :class:`~e2m2e.core.forces.VariableMassFiniteBurn` 的常量推力语义对齐：
    打靶求解器在各段用满油门或部分油门（``throttle ∈ [0,1]``）施加推力。

    Args:
        t_max: 最大推力（N）。
        isp: 比冲（s）。
    """

    t_max: float
    isp: float

    def __post_init__(self) -> None:
        if self.t_max <= 0.0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        if self.isp <= 0.0:
            raise ValueError(f"isp must be positive, got {self.isp}")


@dataclass(frozen=True)
class LowThrustSegment:
    """单段常量控制：throttle + 惯性系方向（归一化后的单位向量）。

    Args:
        throttle: 推力幅值系数，``∈ [0, 1]``。
        direction: 推力方向单位向量，惯性系，``(3,)``。
    """

    throttle: float
    direction: npt.NDArray[np.floating]


@dataclass
class LowThrustShootingSolution:
    """低推力打靶求解结果。

     对齐 :class:`~e2m2e.transfer.config.TransferSolution` 风格，额外携带控制
    历史与 7D 状态（含质量剖面）。

     Attributes:
         time: 采样时间序列，``(M,)``，SPICE et 秒。
         states: 状态序列，``(M, 7)``，``[x, y, z, vx, vy, vz, m]``。
         segments: 各段常量控制。
         final_mass: 末态质量（kg）。
         fuel_consumed: 燃料消耗（kg），``= m0 - final_mass``。
         converged: SLSQP 是否收敛。
         n_iter: SLSQP 迭代次数。
         message: SLSQP 状态消息。
    """

    time: npt.NDArray[np.floating]
    states: npt.NDArray[np.floating]
    segments: tuple[LowThrustSegment, ...]
    final_mass: float
    fuel_consumed: float
    converged: bool
    n_iter: int
    message: str


class LowThrustShooting:
    """单弧多段低推力直接打靶求解器（min-fuel）。

    固定初态与目标末态位置/速度、固定飞行时间，以各段常量控制
    ``(throttle, 方向)`` 为决策变量，最小化燃料消耗（最大化末态质量）。
    传播接龙复用地基 :func:`propagate_compiled_lowthrust`（7D 受控动力学）。

    Args:
        system: 动力学系统，提供 ``coordinate_system`` 与 ``origin``。
        forces: 非推力力模型列表（重力等）。各 force 须支持
            ``to_rust_spec``（否则求解器在构造时抛错）。
        engine: 推进配置（最大推力、比冲）。
        initial_state: 出发状态 ``[r, v]``，``(6,)``，km / km/s。
        initial_mass: 初始质量（kg）。
        target_state: 目标末态 ``[r, v]``，``(6,)``，km / km/s。
        t0: 起始时刻（SPICE et 秒）。
        tf: 终止时刻（SPICE et 秒）。
    """

    def __init__(
        self,
        system: System,
        forces: Sequence[PhysicalModel],
        engine: EngineConfig,
        initial_state: npt.ArrayLike,
        initial_mass: float,
        target_state: npt.ArrayLike,
        t0: float,
        tf: float,
    ) -> None:
        self._system = system
        self._engine = engine
        self._t0 = float(t0)
        self._tf = float(tf)
        if self._tf <= self._t0:
            raise ValueError(f"tf ({self._tf}) must be > t0 ({self._t0})")

        self._initial_state = np.asarray(initial_state, dtype=float)
        if self._initial_state.shape != (6,):
            raise ValueError(f"initial_state must have shape (6,), got {self._initial_state.shape}")
        self._initial_mass = float(initial_mass)
        if self._initial_mass <= 0.0:
            raise ValueError(f"initial_mass must be positive, got {self._initial_mass}")
        self._target_state = np.asarray(target_state, dtype=float)
        if self._target_state.shape != (6,):
            raise ValueError(f"target_state must have shape (6,), got {self._target_state.shape}")

        # 校验力模型支持 Rust 路径，并预序列化为 force 元组
        self._forces_py: list[tuple] = []
        for f in forces:
            spec = f.to_rust_spec(system)
            if spec is None:
                raise NotImplementedError(
                    f"force {f.__class__.__name__} lacks to_rust_spec; "
                    "low-thrust shooting requires the Rust propagation path"
                )
            self._forces_py.append(spec)

        self._observer = getattr(system, "origin", "EARTH")

    # ---- 求解 ----

    def solve(
        self,
        n_segments: int,
        *,
        x0: npt.ArrayLike | None = None,
        throttle_bounds: tuple[float, float] = (0.0, 1.0),
        direction_box: float = 1.0,
        ftol: float = 1e-9,
        maxiter: int = 200,
        verbose: bool = False,
    ) -> LowThrustShootingSolution:
        """求解 min-fuel 低推力打靶。

        Args:
            n_segments: 段数 N，``≥ 1``。决策变量总数 = ``4N``。
            x0: 初猜决策向量 ``(4N,)``；None 时 throttle 全满、方向沿初速方向。
            throttle_bounds: throttle 上下界，默认 ``(0, 1)``。
            direction_box: 方向分量盒约束半宽，默认 1.0（即 ``[-1, 1]``）。
            ftol: SLSQP 目标容差。
            maxiter: SLSQP 最大迭代次数。
            verbose: 是否打印 SLSQP 迭代信息。

        Returns:
            :class:`LowThrustShootingSolution`。
        """
        if n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {n_segments}")

        n_var = 4 * n_segments
        y0 = self._default_x0(n_segments) if x0 is None else np.asarray(x0, dtype=float)
        if y0.shape != (n_var,):
            raise ValueError(f"x0 must have shape ({n_var},), got {y0.shape}")

        lb, ub = self._bounds(n_segments, throttle_bounds, direction_box)

        result = minimize(
            self._fuel_objective,
            y0,
            method="SLSQP",
            bounds=Bounds(lb, ub),
            constraints=[{"type": "eq", "fun": self._terminal_constraint}],
            options={"ftol": ftol, "maxiter": maxiter, "disp": verbose},
        )

        return self._build_solution(
            result.x,
            converged=bool(result.success),
            n_iter=int(result.nit),
            message=str(result.message),
        )

    # ---- 内部：决策向量解码与传播接龙 ----

    def _default_x0(self, n_segments: int) -> np.ndarray:
        """默认初猜：throttle 全满，方向沿初速方向。"""
        v0 = self._initial_state[3:6]
        v_hat = v0 / np.linalg.norm(v0)
        seg = np.concatenate([[1.0], v_hat])
        return np.tile(seg, n_segments)

    def _bounds(
        self,
        n_segments: int,
        throttle_bounds: tuple[float, float],
        direction_box: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """组装决策变量上下界：[throttle_i, ux_i, uy_i, uz_i] × N。"""
        t_lo, t_hi = throttle_bounds
        d = float(direction_box)
        seg_lb = np.array([t_lo, -d, -d, -d])
        seg_ub = np.array([t_hi, d, d, d])
        return np.tile(seg_lb, n_segments), np.tile(seg_ub, n_segments)

    def _decode_segments(
        self, y: npt.NDArray[np.floating]
    ) -> list[tuple[float, float, float, float]]:
        """决策向量 -> 各段 (throttle, 归一化方向) 列表。

        方向向量归一化；零方向（throttle=0 时允许）给占位单位向量。
        """
        flat = np.asarray(y, dtype=float).reshape(-1, 4)
        segs: list[tuple[float, float, float, float]] = []
        for row in flat:
            throttle = float(np.clip(row[0], 0.0, 1.0))
            d = row[1:4]
            norm = np.linalg.norm(d)
            if norm < 1e-15:
                ux, uy, uz = 1.0, 0.0, 0.0  # 占位；throttle=0 时方向不影响
            else:
                ux, uy, uz = d / norm
            segs.append((throttle, ux, uy, uz))
        return segs

    def _propagate_chain(
        self, y: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """接龙传播：返回合并时间序列 (M,) 与 7D 状态序列 (M, 7)。"""
        from e2m2e._integrators import RkMethod, propagate_compiled_lowthrust

        segs = self._decode_segments(y)
        n_segments = len(segs)
        dt = (self._tf - self._t0) / n_segments
        node_times = self._t0 + np.arange(n_segments + 1) * dt

        times_list: list[float] = [float(node_times[0])]
        states_list: list[np.ndarray] = [
            np.concatenate([self._initial_state, [self._initial_mass]])
        ]

        state = np.concatenate([self._initial_state, [self._initial_mass]])
        t_max = self._engine.t_max
        isp = self._engine.isp

        for i, (throttle, ux, uy, uz) in enumerate(segs):
            ti, tip1 = float(node_times[i]), float(node_times[i + 1])
            res = propagate_compiled_lowthrust(
                RkMethod.PD45,
                ti,
                state.tolist(),
                self._estimate_h(state, dt),
                1e-10,
                [ti, tip1],
                self._observer,
                self._forces_py,
                (t_max, isp, throttle, ux, uy, uz),
                500_000,
            )
            seg_states = np.asarray(res["states"], dtype=float)
            # 取段末状态作为下一段初态
            state = seg_states[-1].copy()
            # 合并：跳过段首（与上段段末重复）
            for k in range(1, len(seg_states)):
                times_list.append(float(res["time"][k]))
                states_list.append(seg_states[k])

        return np.asarray(times_list), np.asarray(states_list)

    def _estimate_h(self, state: npt.NDArray[np.floating], dt: float) -> float:
        """段内初始步长估计：取段长的 1/10，并夹到合理区间。"""
        h = dt / 10.0
        return float(np.clip(h, 1.0, dt))

    def _fuel_objective(self, y: npt.NDArray[np.floating]) -> float:
        """目标：最小化 -m_f（最大化末态质量 = min-fuel）。"""
        _, states = self._propagate_chain(y)
        return -float(states[-1][6])

    def _terminal_constraint(self, y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """等式约束：末态 [r, v] - target_state，归一化到 O(1) 量级。

        位置残差除以参考长度、速度残差除以参考速度，使约束与决策变量
        （throttle/方向 ~O(1)）、目标（−mf）尺度匹配，改善 SLSQP 数值差分
        雅可比的性态。SLSQP 默认差分步长 ~1e-8 对未归一化的 km 级位置残差
        过小，雅可比被数值噪声淹没导致不收敛。
        """
        _, states = self._propagate_chain(y)
        r_ref = float(np.linalg.norm(self._initial_state[:3]))
        v_ref = float(np.linalg.norm(self._initial_state[3:6]))
        residual = states[-1][:6] - self._target_state
        return np.concatenate([residual[:3] / r_ref, residual[3:6] / v_ref])

    def _build_solution(
        self,
        y: npt.NDArray[np.floating],
        *,
        converged: bool,
        n_iter: int,
        message: str,
    ) -> LowThrustShootingSolution:
        """从决策向量构造解：再传播一次拿完整轨迹，组装控制历史。"""
        times, states = self._propagate_chain(y)
        segs_decoded = self._decode_segments(y)
        segments = tuple(
            LowThrustSegment(throttle=t, direction=np.array([ux, uy, uz]))
            for (t, ux, uy, uz) in segs_decoded
        )
        final_mass = float(states[-1][6])
        return LowThrustShootingSolution(
            time=times,
            states=states,
            segments=segments,
            final_mass=final_mass,
            fuel_consumed=self._initial_mass - final_mass,
            converged=converged,
            n_iter=n_iter,
            message=message,
        )
