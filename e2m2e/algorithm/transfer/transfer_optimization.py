"""DRO→RO 转移轨道 NLP 优化的高层编排。

实现论文 Cui et al. (2025) "搜索-优化"两步法中的优化阶段：

- 优化变量：``y = (α, T, t_ins)``
- 目标函数：``J(y) = Δv1 + Δv2``
- 约束：位置连续性、速度平行性（或松弛形式）、撞星规避

本模块只承载高层编排：构造优化器、计算目标/约束、组装结果。
SciPy SLSQP 与 COPT 两种求解后端分别封装在
:mod:`e2m2e.transfer.nlp_scipy` 与 :mod:`e2m2e.transfer.nlp_copt`，
由 :class:`DROTRONLPOptimizer` 调度。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ...data.templates import ConvergenceState, FailureCause
from ...data.templates.enums import TransferType
from ...data.types.orbit import Orbit
from ..dynamics import CR3BP_Dynamics, CR3BP_System
from .config import TransferConfig, TransferOptimizationResult

# 后端以模块级 re-export 形式暴露，保持 ``e2m2e.transfer.transfer_optimization.coptpy``
# 等既有 API 兼容（外部模块通过 ``_HAVE_COPT = transfer_optimization.coptpy is not None`` 判定）。
from .nlp_copt import coptpy, optimize_with_copt  # noqa: F401
from .nlp_core import NLPOptimizationVariables
from .propulsion import ImpulsivePropulsion
from .terminal import OrbitTerminal, TerminalCondition


class DROTRONLPOptimizer:
    """DRO→RO 转移轨道 NLP 优化器

    实现论文 Section III.B 的优化阶段算法。
    默认通过 SciPy SLSQP 求解；调用方可改走 COPT 后端
    （见 :func:`~e2m2e.transfer.nlp_copt.optimize_with_copt`）。

    Attributes:
        system: CR3BP 系统对象
        dynamics: CR3BP 动力学对象
        departure_orbit: 出发点轨道
        arrival_orbit: 目标轨道
        departure_state: 出发点状态
        alpha_range: α 搜索范围
        velocity_angle_tolerance: 速度平行性容差（弧度）
        earth_radius: 地球半径（无量纲）
        moon_radius: 月球半径（无量纲）
    """

    # 优化变量的默认搜索范围
    DEFAULT_ALPHA_RANGE = (0.5, 2.5)
    DEFAULT_TRANSFER_TIME_RANGE = (1.0, 30.0)
    DEFAULT_T_INS_RANGE = (0.0, 10.0)

    # 地月天体碰撞检测半径（无量纲单位）
    EARTH_RADIUS_ND = 1.0 / 389703.0 * 6378.137  # 地球半径 / 地月距离
    MOON_RADIUS_ND = 1738.1 / 384400.0  # 月球半径 / 地月距离

    DEFAULT_VELOCITY_ANGLE_TOL = 1e-6

    def __init__(
        self,
        system: CR3BP_System,
        dynamics: CR3BP_Dynamics,
        *,
        departure_terminal: TerminalCondition | None = None,
        arrival_terminal: TerminalCondition | None = None,
        departure_orbit: Orbit | None = None,
        arrival_orbit: Orbit | None = None,
        departure_state: np.ndarray | None = None,
        config: TransferConfig | None = None,
        propulsion: ImpulsivePropulsion | None = None,
    ):
        """初始化 NLP 优化器（issue #161）。

        支持两种构造路径（不可混用）：

        * **新接口（推荐）** — ``(departure_terminal, arrival_terminal)``，
          内部走 :class:`TerminalCondition` 接口；这是 :class:`StateTerminal`
          等非轨道型终端接入的唯一方式。
        * **旧接口（向后兼容）** — ``(departure_orbit, arrival_orbit,
          departure_state)``，内部等价于把两条轨道包成 :class:`OrbitTerminal`。
        """
        self.system = system
        self.dynamics = dynamics
        self.mu = system.mu
        self.propulsion = propulsion if propulsion is not None else ImpulsivePropulsion()

        # ``departure_state`` 在两种接口下都允许（新接口下作覆盖），
        # 因此混用判定只看 ``departure_orbit``/``arrival_orbit``。
        has_terminals = departure_terminal is not None or arrival_terminal is not None
        has_legacy = departure_orbit is not None or arrival_orbit is not None
        if has_terminals and has_legacy:
            raise ValueError("DROTRONLPOptimizer: cannot mix new and legacy interfaces")
        if has_terminals:
            if departure_terminal is None or arrival_terminal is None:
                raise ValueError("DROTRONLPOptimizer: both terminals required")
            self.departure_terminal: TerminalCondition = departure_terminal
            self.arrival_terminal: TerminalCondition = arrival_terminal
        elif has_legacy:
            if departure_orbit is None or arrival_orbit is None:
                raise ValueError("DROTRONLPOptimizer: legacy interface requires both orbits")
            self.departure_terminal = OrbitTerminal(departure_orbit)
            self.arrival_terminal = OrbitTerminal(arrival_orbit)
        else:
            raise ValueError("DROTRONLPOptimizer: must provide terminal pair or orbit pair")

        # 实际出发点状态：显式覆盖 > 终端默认首点
        self._departure_state = (
            np.asarray(departure_state, dtype=float)
            if departure_state is not None
            else self.departure_terminal.get_initial_state()
        )

        # 兼容历史读取（仅 OrbitTerminal 提供 .orbit）
        self.departure_state = self._departure_state
        self.departure_orbit = getattr(self.departure_terminal, "orbit", None)
        self.arrival_orbit = getattr(self.arrival_terminal, "orbit", None)

        self.alpha_range = (
            config.nlp_alpha_range if config is not None else self.DEFAULT_ALPHA_RANGE
        )
        self.transfer_time_range = self.DEFAULT_TRANSFER_TIME_RANGE
        self.t_ins_range = (
            config.nlp_t_ins_range
            if config is not None and config.nlp_t_ins_range is not None
            else self.DEFAULT_T_INS_RANGE
        )

        self.velocity_angle_tol = (
            config.nlp_velocity_angle_tol if config is not None else self.DEFAULT_VELOCITY_ANGLE_TOL
        )

        self.earth_radius = config.nlp_earth_radius if config is not None else self.EARTH_RADIUS_ND
        self.moon_radius = config.nlp_moon_radius if config is not None else self.MOON_RADIUS_ND

        self._use_relaxed_velocity = (
            config.nlp_use_relaxed_velocity if config is not None else False
        )
        self._verbose = config.nlp_verbose if config is not None else True

        self._last_trajectory: tuple[np.ndarray, np.ndarray] | None = None
        # 最近一次传播的 failure 标记（ADR 0020 决策 2）：forward_integrate 写入，
        # _evaluate_all 读取，替代下游 len(states)==0 嗅探。
        self._last_prop_status: ConvergenceState = ConvergenceState.CONVERGED
        self._last_prop_cause: FailureCause = FailureCause.NONE
        self._progress_callback: Callable | None = None
        self._eval_cache: dict[str, Any] | None = None
        self._eval_cache_key: bytes | None = None
        self._cache_enabled: bool = False

    @classmethod
    def from_orbits(
        cls,
        system: CR3BP_System,
        dynamics: CR3BP_Dynamics,
        departure_orbit: Orbit,
        arrival_orbit: Orbit,
        departure_state: np.ndarray | None = None,
        config: TransferConfig | None = None,
        propulsion: ImpulsivePropulsion | None = None,
    ) -> DROTRONLPOptimizer:
        """通过 ``Orbit`` 直接构造（向后兼容的便捷类方法）。

        内部等价于把两条轨道包成 :class:`OrbitTerminal` 再走新接口。

        Args:
            system: CR3BP 系统对象
            dynamics: CR3BP 动力学对象
            departure_orbit: 出发轨道
            arrival_orbit: 到达轨道
            departure_state: 出发点状态；``None`` 时取轨道首点
            config: 优化配置
            propulsion: 推进模型

        Returns:
            ``DROTRONLPOptimizer`` 实例
        """
        # ``departure_state`` 在显式传入时通过关键字传给 ``__init__``，
        # 否则让 ``__init__`` 用 ``OrbitTerminal.get_initial_state()``，
        # 这样 ``__init__`` 不会因 ``has_legacy=...departure_state...`` 误判接口。
        if departure_state is not None:
            return cls(
                system=system,
                dynamics=dynamics,
                departure_terminal=OrbitTerminal(departure_orbit),
                arrival_terminal=OrbitTerminal(arrival_orbit),
                departure_state=departure_state,
                config=config,
                propulsion=propulsion,
            )
        return cls(
            system=system,
            dynamics=dynamics,
            departure_terminal=OrbitTerminal(departure_orbit),
            arrival_terminal=OrbitTerminal(arrival_orbit),
            config=config,
            propulsion=propulsion,
        )

    def set_progress_callback(self, callback: Callable | None) -> None:
        """设置迭代进度回调函数。

        Args:
            callback: 签名 ``(iteration, obj, alpha, T, t_ins) -> None``，
                或 ``None`` 清除回调。
        """
        self._progress_callback = callback

    def enable_cache(self, enabled: bool = True) -> None:
        """启用/禁用 ``_evaluate_all`` 结果缓存，避免同一点重复积分。

        Args:
            enabled: ``True`` 开启缓存。
        """
        self._cache_enabled = enabled

    def _y_key(self, y: np.ndarray) -> bytes:
        """将优化变量序列化为字节键，用于缓存匹配。"""
        return y.tobytes()

    def _evaluate_all(self, y: np.ndarray) -> dict[str, Any]:
        """一次性计算目标函数和所有约束，结果缓存以避免重复积分。

        Args:
            y: 优化变量 ``[alpha, transfer_time, t_ins]``。

        Returns:
            包含 ``objective``、``pos_violation``、``cos_angle`` 等键的字典。
        """
        key = self._y_key(y)
        if self._cache_enabled and self._eval_cache_key == key and self._eval_cache is not None:
            return self._eval_cache

        alpha, transfer_time, t_ins = y
        dep_state = self._departure_state

        v_injection = self.compute_departure_velocity(dep_state, alpha)
        initial_state = np.concatenate([dep_state[:3], v_injection])

        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        # 显式 failure 标记（ADR 0020 决策 2）：读 status 而非嗅探 len(states)==0。
        empty = self._last_prop_status is not ConvergenceState.CONVERGED

        final_state = states[-1] if not empty else np.zeros(6)

        # 到达状态走 TerminalCondition 接口
        if not empty:
            ins_pos, ins_vel = self.arrival_terminal.get_arrival_state(float(t_ins), self.dynamics)
            insertion_state = np.concatenate([ins_pos, ins_vel])
        else:
            insertion_state = np.zeros(6)

        if not empty:
            cost = self.propulsion.compute_cost(
                dep_state,
                v_injection,
                final_state[3:],
                insertion_state[3:],
            )
            dv1 = cost.dv1
            dv2 = cost.dv2
        else:
            # 传播失败无 Δv 值：不再用 1e10 惩罚污染目标（ADR 0020 决策 2），
            # 不可行由约束冲突（pos_violation/vel_constraint）与 INFEASIBLE 表达。
            dv1 = float("nan")
            dv2 = float("nan")
            cost = None

        pos_diff = final_state[:3] - insertion_state[:3]
        pos_violation = np.dot(pos_diff, pos_diff) if not empty else 1e6

        v_f = final_state[3:]
        v_ins = insertion_state[3:]
        v_f_norm = np.linalg.norm(v_f)
        v_ins_norm = np.linalg.norm(v_ins)
        if not empty and v_f_norm > 1e-10 and v_ins_norm > 1e-10:
            # cos_angle 接近 1 表示两速度几乎同向（理想情况）
            cos_angle = np.dot(v_f, v_ins) / (v_f_norm * v_ins_norm)
        else:
            # cos_angle = -1.0 标记"反向"惩罚，优化器会避开此区域
            cos_angle = -1.0

        cache = {
            "v_injection": v_injection,
            "times": times,
            "states": states,
            "final_state": final_state,
            "insertion_state": insertion_state,
            "dv1": dv1,
            "dv2": dv2,
            # 传播失败目标为 inf（不可行域），不再用 2e10 惩罚值污染目标。
            "objective": cost.total if cost is not None else float("inf"),
            # 约束冲突保留有限大值 1e6：不可行信号，供优化器识别；非目标惩罚。
            "pos_violation": pos_violation if not empty else 1e6,
            "cos_angle": cos_angle,
            "vel_constraint": cos_angle - 1.0 if not empty else 1e6,
            # 传播层 DIVERGED 在 NLP 层规范化为 INFEASIBLE；cause 透传诊断来源。
            "status": ConvergenceState.INFEASIBLE if empty else ConvergenceState.CONVERGED,
            "cause": self._last_prop_cause if empty else FailureCause.NONE,
            "empty": empty,
        }

        if self._cache_enabled:
            self._eval_cache = cache
            self._eval_cache_key = key

        return cache

    def compute_departure_velocity(
        self, state: np.ndarray, alpha: float, beta: float = 0.0
    ) -> np.ndarray:
        """根据 α 和 β 计算出发注入速度。

        委托给 ``self.propulsion.compute_departure_velocity``，保留方法签名以兼容外部调用。

        Args:
            state: 出发点状态 ``[x, y, z, vx, vy, vz]``
            alpha: 切向速度比（缩放切向分量）
            beta: 法向速度比（缩放法向分量），默认 0.0（纯切向）

        Returns:
            注入速度向量 ``[vx, vy, vz]``
        """
        return self.propulsion.compute_departure_velocity(state, alpha=alpha, beta=beta)

    def forward_integrate(
        self,
        initial_state: np.ndarray,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """前向积分转移弧

        Args:
            initial_state: 初始状态 ``[x, y, z, vx, vy, vz]``
            t_span: 积分时间范围 ``(t0, tf)``
            t_eval: 评估时间点

        Returns:
            ``(times, states)``：时间序列和状态序列
        """
        if t_eval is None:
            step = max(0.01, self.dynamics.max_step)
            n_steps = int((t_span[1] - t_span[0]) / step) + 1
            t_eval = np.linspace(t_span[0], t_span[1], n_steps)

        result = self.dynamics.propagate(
            initial_state=initial_state,
            t_span=t_span,
            t_eval=t_eval,
            with_stm=False,
            with_jacobi=False,
        )

        times = result["time"]
        states = result["states"]

        # 透传传播 failure 标记（ADR 0020 决策 2）：读 status/cause 而非下游
        # len==0 嗅探。对未返回标记的传播实现做兜底：空 states 即视为 DIVERGED。
        status = result.get("status", ConvergenceState.CONVERGED)
        cause = result.get("cause", FailureCause.NONE)
        if states.shape[0] == 0 and status is ConvergenceState.CONVERGED:
            status = ConvergenceState.DIVERGED
            cause = FailureCause.DIVERGENCE_DETECTED
        self._last_prop_status = status
        self._last_prop_cause = cause

        self._last_trajectory = (times, states)

        return times, states

    def get_arrival_state_at_t_ins(self, t_ins: float) -> tuple[np.ndarray, np.ndarray]:
        """获取目标轨道上 t_ins（绝对时间）对应的状态

        Args:
            t_ins: 绝对时间（与 ``orbit.times`` 同一坐标系）

        Returns:
            ``(position, velocity)``：位置和速度
        """
        return self.arrival_terminal.get_arrival_state(float(t_ins), self.dynamics)

    def objective_function(self, y: np.ndarray) -> float:
        """目标函数 ``J(y) = Δv1 + Δv2``

        Args:
            y: 优化变量 ``[alpha, T, t_ins]``

        Returns:
            总脉冲代价
        """
        cache = self._evaluate_all(y)
        if cache["empty"]:
            # 传播失败目标为 inf（ADR 0020 决策 2），不再用 1e10 惩罚污染目标；
            # 不可行由约束冲突识别。
            return float("inf")
        return cache["objective"]

    def constraint_position(self, y: np.ndarray) -> float:
        """位置连续性约束 Eq.(13)

        ``(x_f - x_ins)^2 + (y_f - y_ins)^2 + (z_f - z_ins)^2 = 0``

        Args:
            y: 优化变量 ``[alpha, T, t_ins]``

        Returns:
            约束违反量
        """
        cache = self._evaluate_all(y)
        return cache["pos_violation"]

    def constraint_velocity_parallel(self, y: np.ndarray) -> float:
        """速度平行性约束 Eq.(14) 或松弛 Eq.(17)

        ``v_f · v_ins / (||v_f|| ||v_ins||) - 1 = 0``

        Args:
            y: 优化变量 ``[alpha, T, t_ins]``

        Returns:
            约束违反量
        """
        cache = self._evaluate_all(y)
        return cache["vel_constraint"]

    def check_collision(self, y: np.ndarray) -> tuple[bool, bool]:
        """检查是否撞击地球或月球

        Args:
            y: 优化变量 ``[alpha, T, t_ins]``

        Returns:
            ``(earth_collision, moon_collision)``：是否撞击地球、月球
        """
        alpha, transfer_time, t_ins = y
        dep_state = self._departure_state

        v_injection = self.compute_departure_velocity(dep_state, alpha)
        initial_state = np.concatenate([dep_state[:3], v_injection])

        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        if len(states) == 0:
            # 传播失败（轨迹为空）无法判断是否碰撞：不再谎报无碰撞（#352），
            # 让调用方把该候选计为不可行。
            from ...exceptions import PropagationFailure

            raise PropagationFailure("check_collision: 轨迹为空（传播失败），无法判断是否碰撞")

        earth_collision = False
        moon_collision = False

        for state in states:
            pos = state[:3]

            r_earth = np.sqrt((pos[0] + self.mu) ** 2 + pos[1] ** 2 + pos[2] ** 2)
            if r_earth < self.earth_radius:
                earth_collision = True

            r_moon = np.sqrt((pos[0] - 1 + self.mu) ** 2 + pos[1] ** 2 + pos[2] ** 2)
            if r_moon < self.moon_radius:
                moon_collision = True

        return earth_collision, moon_collision

    def optimize(
        self,
        initial_guess: NLPOptimizationVariables | None = None,
        alpha_range: tuple[float, float] | None = None,
        transfer_time_range: tuple[float, float] | None = None,
        t_ins_range: tuple[float, float] | None = None,
        use_relaxed_velocity_constraint: bool | None = None,
        velocity_angle_constraint: float | None = None,
        verbose: bool | None = None,
    ) -> TransferOptimizationResult:
        """执行 SciPy 后端的 NLP 优化。

        委托给 :func:`e2m2e.transfer.nlp_scipy.solve_with_scipy`。完整参数说明见该函数。
        """
        from .nlp_scipy import solve_with_scipy

        return solve_with_scipy(
            self,
            initial_guess=initial_guess,
            alpha_range=alpha_range,
            transfer_time_range=transfer_time_range,
            t_ins_range=t_ins_range,
            use_relaxed_velocity_constraint=use_relaxed_velocity_constraint,
            velocity_angle_constraint=velocity_angle_constraint,
            verbose=verbose,
        )

    def _compute_cos_angle(self, y: np.ndarray) -> float:
        """计算速度夹角余弦"""
        cache = self._evaluate_all(y)
        return cache["cos_angle"]

    def _build_result(
        self,
        variables: NLPOptimizationVariables,
        status: ConvergenceState,
        cause: FailureCause,
        message: str,
        use_relaxed_constraint: bool = False,
        velocity_angle_constraint: float = 0.0,
    ) -> TransferOptimizationResult:
        """构建优化结果对象"""
        alpha = variables.alpha
        transfer_time = variables.transfer_time
        t_ins = variables.t_ins
        dep_state = self._departure_state

        v_injection = self.compute_departure_velocity(dep_state, alpha)
        initial_state = np.concatenate([dep_state[:3], v_injection])
        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        # 到达状态走 TerminalCondition 接口
        ins_pos, ins_vel = self.arrival_terminal.get_arrival_state(float(t_ins), self.dynamics)
        insertion_state = np.concatenate([ins_pos, ins_vel])
        final_state = states[-1] if len(states) > 0 else None

        cost = self.propulsion.compute_cost(
            dep_state,
            v_injection,
            final_state[3:] if final_state is not None else np.zeros(3),
            insertion_state[3:],
        )

        violation = {}
        if status is ConvergenceState.CONVERGED:
            violation["position"] = self.constraint_position(variables.to_array())
            if use_relaxed_constraint:
                violation["velocity"] = max(
                    0,
                    np.cos(velocity_angle_constraint)
                    - self._compute_cos_angle(variables.to_array()),
                )
            else:
                violation["velocity"] = abs(self.constraint_velocity_parallel(variables.to_array()))

        max_violation = max(violation.values()) if violation else 0.0

        transfer_type = self._classify_transfer(transfer_time, times, states, insertion_state)

        return TransferOptimizationResult(
            status=status,
            cause=cause,
            message=message,
            departure_state=dep_state.copy(),
            departure_alpha=alpha,
            departure_beta=0.0,
            insertion_state=insertion_state,
            final_state=final_state,
            delta_v1=cost.dv1,
            delta_v2=cost.dv2,
            total_delta_v=cost.total,
            transfer_time=transfer_time,
            t_ins=t_ins,
            transfer_trajectory=states,
            transfer_trajectory_times=times,
            constraints_violation=max_violation,
            transfer_type=transfer_type,
        )

    def _classify_transfer(
        self,
        transfer_time: float,
        times: np.ndarray,
        states: np.ndarray,
        insertion_state: np.ndarray,
    ) -> TransferType:
        """根据转移时间与轨迹几何范围分类转移类型。

        分类规则（启发式阈值，单位均为无量纲）：

        - ``transfer_time < 20.0`` 且 ``max(x) < 1.5``：直达转移 ``TransferType.DIRECT``
        - ``max(x) > 3.0``：轨迹绕到地月系统外侧，外部转移 ``TransferType.EXTERNAL``
        - 其余情况：含月球引力辅助的转移 ``TransferType.LGA``

        积分失败或轨迹为空时返回 :class:`TransferType.UNKNOWN`（#352：
        不再假装 DIRECT——空轨迹无法分类）。

        该结果会写入 :class:`TransferOptimizationResult.transfer_type`，
        可视化层（``plot_solution_plane(..., color_by="transfer_type")``）
        据此按类型着色，因此本方法属于活跃路径，并非已废弃。

        Args:
            transfer_time: 转移时间（无量纲时间）
            times: 轨迹时间序列
            states: 轨迹状态序列，形状 ``(n_steps, 6)``
            insertion_state: 插入点状态

        Returns:
            转移类型枚举。
        """
        if len(states) == 0:
            return TransferType.UNKNOWN

        x_max_traj = np.max(states[:, 0])

        # 20.0 无量纲时间 ≈ 地月系中短时间转移阈值
        # 1.5 无量纲距离 ≈ 月球轨道半径（地月距离单位），限制轨迹在月球以内
        if transfer_time < 20.0 and x_max_traj < 1.5:
            return TransferType.DIRECT

        # 3.0 无量纲距离 ≈ 远超月球轨道，表明轨迹绕到地月系统外侧
        if x_max_traj > 3.0:
            return TransferType.EXTERNAL

        return TransferType.LGA


def optimize_transfer(
    system: CR3BP_System,
    dynamics: CR3BP_Dynamics,
    departure_orbit: Orbit,
    arrival_orbit: Orbit,
    departure_state: np.ndarray,
    initial_guess: NLPOptimizationVariables | None = None,
    propulsion: ImpulsivePropulsion | None = None,
    **kwargs,
) -> TransferOptimizationResult:
    """便捷函数: 优化 DRO→RO 转移（SciPy SLSQP 后端）

    Args:
        system: CR3BP 系统
        dynamics: CR3BP 动力学
        departure_orbit: 出发点轨道
        arrival_orbit: 目标轨道
        departure_state: 出发点状态
        initial_guess: 初始猜测
        propulsion: 推进模型；``None`` 时使用 ``ImpulsivePropulsion()``
        **kwargs: 其他优化参数（透传给 ``optimizer.optimize``）

    Returns:
        优化结果
    """
    optimizer = DROTRONLPOptimizer(
        system=system,
        dynamics=dynamics,
        departure_orbit=departure_orbit,
        arrival_orbit=arrival_orbit,
        departure_state=departure_state,
        propulsion=propulsion,
    )

    return optimizer.optimize(initial_guess=initial_guess, **kwargs)
