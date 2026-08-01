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

from ..forces import PhysicalModel

if TYPE_CHECKING:
    from ..dynamics import System


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

    def solve_from_qlaw(
        self,
        n_segments: int,
        target_oe: tuple[float, float, float],
        forces: Sequence[PhysicalModel],
        *,
        step: float = 120.0,
        use_analytic_jac: bool = True,
        maxiter: int = 100,
        verbose: bool = False,
    ) -> LowThrustShootingSolution:
        """用 Q-law 生成初猜，再解析雅可比打磨。

         两级流程（gap-analysis）：Q-law 前向反馈积分产出次优控制历史
        （:func:`~e2m2e.transfer.qlaw.qlaw_guess`），喂 :meth:`solve` 做
         min-fuel 最优控制打磨。Q-law 解决「满推力初猜推过头」的发散问题。

         Args:
             n_segments: 段数 N（Q-law 重采样 + 求解器决策变量数 = 3N）。
             target_oe: Q-law 目标 ``(a_T, e_T, i_T)``（只控 a,e,i）。
             forces: 非推力力模型（与构造时一致，Q-law 用于查 μ）。
             step: Q-law 前向积分步长（秒）。
             use_analytic_jac: 打磨阶段用解析雅可比。
             maxiter: SLSQP 最大迭代。
             verbose: 打印进度。
        """
        from .qlaw import qlaw_guess

        y0, _segments, _qh, _final = qlaw_guess(
            self._system,  # type: ignore[arg-type]
            forces,
            self._engine,
            self._initial_state,
            self._initial_mass,
            target_oe,
            self._t0,
            self._tf,
            n_segments,
            step=step,
            verbose=verbose,
        )
        return self.solve(
            n_segments, x0=y0, use_analytic_jac=use_analytic_jac, maxiter=maxiter, verbose=verbose
        )

    def solve(
        self,
        n_segments: int,
        *,
        x0: npt.ArrayLike | None = None,
        throttle_bounds: tuple[float, float] = (0.0, 1.0),
        use_analytic_jac: bool = True,
        ftol: float = 1e-9,
        maxiter: int = 200,
        verbose: bool = False,
    ) -> LowThrustShootingSolution:
        """求解 min-fuel 低推力打靶。

        Args:
            n_segments: 段数 N，``≥ 1``。决策变量每段 ``(throttle, θ₁, θ₂)``，
                总数 ``3N``（角度参数化方向，Du 2024 式 5）。
            x0: 初猜决策向量 ``(3N,)``；None 时 throttle 全满、方向角对齐初速。
            throttle_bounds: throttle 上下界，默认 ``(0, 1)``。
            use_analytic_jac: True 时用解析雅可比（灵敏度方程，每迭代 1 次传播）；
                False 回退 SLSQP 数值差分（每迭代 3N+1 次传播）。
            ftol: SLSQP 目标容差。
            maxiter: SLSQP 最大迭代次数。
            verbose: 是否打印 SLSQP 迭代信息。

        Returns:
            :class:`LowThrustShootingSolution`。
        """
        if n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {n_segments}")

        n_var = 3 * n_segments
        y0 = self._default_x0(n_segments) if x0 is None else np.asarray(x0, dtype=float)
        if y0.shape != (n_var,):
            raise ValueError(f"x0 must have shape ({n_var},), got {y0.shape}")

        lb, ub = self._bounds(n_segments, throttle_bounds)

        constraints: dict[str, object] = {"type": "eq", "fun": self._terminal_constraint}
        if use_analytic_jac:
            constraints["jac"] = self._terminal_jacobian

        result = minimize(
            self._fuel_objective,
            y0,
            method="SLSQP",
            jac=self._fuel_jacobian if use_analytic_jac else None,
            bounds=Bounds(lb, ub),
            constraints=[constraints],
            options={"ftol": ftol, "maxiter": maxiter, "disp": verbose},
        )

        return self._build_solution(
            result.x,
            converged=bool(result.success),
            n_iter=int(result.nit),
            message=str(result.message),
        )

    # ---- 内部：决策向量解码与传播接龙 ----

    @staticmethod
    def _angles_to_direction(theta1: float, theta2: float) -> np.ndarray:
        """角度参数化方向向量（Du 2024 式 5）。"""
        return np.array(
            [
                np.cos(theta1) * np.cos(theta2),
                np.sin(theta1) * np.cos(theta2),
                np.sin(theta2),
            ]
        )

    def _default_x0(self, n_segments: int) -> np.ndarray:
        """默认初猜：throttle 全满，方向角对齐初速方向。"""
        v0 = self._initial_state[3:6]
        v_hat = v0 / np.linalg.norm(v0)
        # 反推初速方向的 (θ₁, θ₂)：θ₁=atan2(vy,vx)，θ₂=asin(vz)
        theta1 = float(np.arctan2(v_hat[1], v_hat[0]))
        theta2 = float(np.arcsin(v_hat[2]))
        seg = np.array([1.0, theta1, theta2])
        return np.tile(seg, n_segments)

    def _bounds(
        self,
        n_segments: int,
        throttle_bounds: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """组装决策变量上下界：[throttle_i, θ₁_i, θ₂_i] × N。"""
        t_lo, t_hi = throttle_bounds
        seg_lb = np.array([t_lo, -np.pi, -np.pi / 2])
        seg_ub = np.array([t_hi, np.pi, np.pi / 2])
        return np.tile(seg_lb, n_segments), np.tile(seg_ub, n_segments)

    def _decode_segments(self, y: npt.NDArray[np.floating]) -> list[tuple[float, float, float]]:
        """决策向量 -> 各段 (throttle, θ₁, θ₂) 列表。"""
        flat = np.asarray(y, dtype=float).reshape(-1, 3)
        return [(float(np.clip(row[0], 0.0, 1.0)), float(row[1]), float(row[2])) for row in flat]

    def _propagate_chain(
        self, y: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """接龙传播：返回合并时间序列 (M,) 与 7D 状态序列 (M, 7)。

        用无灵敏度的 ``propagate_compiled_lowthrust``（重建轨迹用）。
        """
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

        for i, (throttle, theta1, theta2) in enumerate(segs):
            ti, tip1 = float(node_times[i]), float(node_times[i + 1])
            alpha = self._angles_to_direction(theta1, theta2)
            res = propagate_compiled_lowthrust(
                RkMethod.PD45,
                ti,
                state.tolist(),
                self._estimate_h(state, dt),
                1e-10,
                [ti, tip1],
                self._observer,
                self._forces_py,
                (t_max, isp, throttle, float(alpha[0]), float(alpha[1]), float(alpha[2])),
                500_000,
            )
            seg_states = np.asarray(res["states"], dtype=float)
            state = seg_states[-1].copy()
            for k in range(1, len(seg_states)):
                times_list.append(float(res["time"][k]))
                states_list.append(seg_states[k])

        return np.asarray(times_list), np.asarray(states_list)

    def _propagate_chain_with_jacobian(
        self, y: npt.NDArray[np.floating]
    ) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """带灵敏度的接龙传播，返回 (末态质量, 末态[r,v] 6D, 全雅可比 6×3N)。

        每段调 ``propagate_compiled_lowthrust_sensitivity``，收末端灵敏度 S(7×3)
        与 STM Φ(6×6)。全局末端对段 i 控制的雅可比 = 复合 STM(段 i+1..N) · S_i
        的前 6 行。复合 STM 从后往前累积。
        """
        from e2m2e._integrators import RkMethod, propagate_compiled_lowthrust_sensitivity

        segs = self._decode_segments(y)
        n_segments = len(segs)
        dt = (self._tf - self._t0) / n_segments
        node_times = self._t0 + np.arange(n_segments + 1) * dt
        t_max = self._engine.t_max
        isp = self._engine.isp

        # 先正向接龙，收集每段末端 state7、Φ(6×6)、S(7×3)
        seg_ends: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        state = np.concatenate([self._initial_state, [self._initial_mass]])
        for i, (throttle, theta1, theta2) in enumerate(segs):
            ti, tip1 = float(node_times[i]), float(node_times[i + 1])
            res = propagate_compiled_lowthrust_sensitivity(
                RkMethod.PD45,
                ti,
                state.tolist(),
                self._estimate_h(state, dt),
                1e-10,
                [ti, tip1],
                self._observer,
                self._forces_py,
                (t_max, isp, throttle, theta1, theta2),
                500_000,
            )
            end_state = np.asarray(res["states"][-1])
            end_stm = np.asarray(res["stm"][-1]).reshape(6, 6)
            end_sens = np.asarray(res["sensitivity"][-1]).reshape(7, 3)
            seg_ends.append((end_state, end_stm, end_sens))
            state = end_state  # 下段初态

        final_state7 = seg_ends[-1][0]
        # 链式雅可比：从后往前累积复合 STM，J_i = Φ_{i+1..N} · S_i（前 6 行）
        jac = np.zeros((6, 3 * n_segments))
        composite_phi = np.eye(6)  # 末端到末端
        for i in range(n_segments - 1, -1, -1):
            end_state_i, end_stm_i, end_sens_i = seg_ends[i]
            # J_i（段 i 控制对全局末端的灵敏度）= composite_phi · S_i[:6, :]
            jac[:, i * 3 : (i + 1) * 3] = composite_phi @ end_sens_i[:6, :]
            # 更新复合 STM：composite_phi = Φ_{i+1..N} → Φ_{i..N} = composite_phi · Φ_i
            composite_phi = composite_phi @ end_stm_i

        final_mass = float(final_state7[6])
        return final_mass, final_state7[:6], jac

    def _estimate_h(self, state: npt.NDArray[np.floating], dt: float) -> float:
        """段内初始步长估计：取段长的 1/10，并夹到合理区间。"""
        h = dt / 10.0
        return float(np.clip(h, 1.0, dt))

    # ---- 目标、约束及其解析雅可比 ----

    def _fuel_objective(self, y: npt.NDArray[np.floating]) -> float:
        """目标：最小化 -m_f（最大化末态质量 = min-fuel）。"""
        _, states = self._propagate_chain(y)
        return -float(states[-1][6])

    def _fuel_jacobian(self, y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """目标解析雅可比：∂(-m_f)/∂y。

        ∂m_f/∂(段 i 控制) = 末态质量对段 i 控制的灵敏度 = 复合 STM 的质量行
        与 S_i 质量行的复合。质量不进 6×6 STM（ṁ 只依赖 throttle），故段 i 之后
        的质量变化只来自段 i 自身的 ∂m_f/∂(段 i throttle) = S_i[6, 0]·dt... 但
        实际上质量是单调累积：末态质量 = 初态 - Σ 燃料消耗，燃料消耗只依赖各段
        throttle 与段长。因此 ∂m_f/∂(段 i throttle) = -T_max·dt/(Isp·g0)，
        ∂m_f/∂θ = 0。
        """
        from e2m2e._integrators import propagate_compiled_lowthrust_sensitivity  # noqa: F401

        g0 = 9.81
        dt = (self._tf - self._t0) / (len(self._decode_segments(y)))
        dm_dthr = -self._engine.t_max * dt / (self._engine.isp * g0)
        jac = np.zeros(len(y))
        for i in range(len(y) // 3):
            jac[i * 3] = -dm_dthr  # ∂(-m_f)/∂throttle = -∂m_f/∂thr = +(T dt/Isp g0)
        return jac

    def _terminal_constraint(self, y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """等式约束：归一化末态 [r, v] - target_state。"""
        _, final_rv, _ = self._propagate_chain_with_jacobian(y)
        r_ref = float(np.linalg.norm(self._initial_state[:3]))
        v_ref = float(np.linalg.norm(self._initial_state[3:6]))
        residual = final_rv - self._target_state
        return np.concatenate([residual[:3] / r_ref, residual[3:6] / v_ref])

    def _terminal_jacobian(self, y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """约束解析雅可比：归一化的 jac = 链式雅可比各行除以参考量。"""
        _, _, jac = self._propagate_chain_with_jacobian(y)
        r_ref = float(np.linalg.norm(self._initial_state[:3]))
        v_ref = float(np.linalg.norm(self._initial_state[3:6]))
        scale = np.array([r_ref, r_ref, r_ref, v_ref, v_ref, v_ref])
        return jac / scale[:, None]

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
            LowThrustSegment(throttle=t, direction=self._angles_to_direction(t1, t2))
            for (t, t1, t2) in segs_decoded
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
