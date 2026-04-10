"""DRO-RO 转移轨迹优化模块

提供基于 NLP 方法（Cui et al. 2025）的简化转移轨迹优化接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np

from ..core.dynamics import CR3BP_Dynamics
from ..core.orbit import Orbit
from ..core.system import CR3BP_System

from . import transfer_optimization
from .transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
    optimize_with_copt,
)

_HAVE_COPT = transfer_optimization.coptpy is not None


DU = 3.84405000e5


@dataclass
class TransferConfig:
    """转移优化配置

    Attributes:
        alpha_min: 切向速度比下界
        alpha_max: 切向速度比上界
        earth_radius: 地球碰撞检测半径（无量纲）
        moon_radius: 月球碰撞检测半径（无量纲）
        use_relaxed_velocity: 是否使用松弛速度约束
        velocity_angle_tol: 松弛速度约束角度容差（弧度）
        use_copt: 是否优先使用 COPT 优化器
        fallback_to_scipy: COPT 失败时是否回退到 SciPy
    """

    alpha_min: float = 0.5
    alpha_max: float = 2.5
    earth_radius: float = 200.0 / DU
    moon_radius: float = 100.0 / DU
    use_relaxed_velocity: bool = True
    velocity_angle_tol: float = 0.05
    use_copt: bool = False
    fallback_to_scipy: bool = True


@dataclass
class TransferOptimizationResult:
    """转移优化结果

    Attributes:
        success: 优化是否成功
        message: 求解器消息
        departure_state: 出发点状态 [x, y, z, vx, vy, vz]
        departure_alpha: 出发点切向速度比
        departure_beta: 出发点法向速度比
        insertion_state: RO 上的插入点状态 [x, y, z, vx, vy, vz]
        final_state: 插入后最终状态 [x, y, z, vx, vy, vz]
        delta_v1: 出发脉冲大小
        delta_v2: 插入脉冲大小
        total_delta_v: 总脉冲（delta_v1 + delta_v2）
        transfer_time: 转移时长
        t_ins: RO 上的插入时间
        transfer_trajectory: 完整转移轨迹 [n_steps, 6]
        transfer_trajectory_times: 轨迹时间序列 [n_steps]
        constraints_violation: 最大约束违反量
    """

    success: bool = False
    message: str = ""
    departure_state: Optional[np.ndarray] = None
    departure_alpha: float = 0.0
    departure_beta: float = 0.0
    insertion_state: Optional[np.ndarray] = None
    final_state: Optional[np.ndarray] = None
    delta_v1: float = 0.0
    delta_v2: float = 0.0
    total_delta_v: float = 0.0
    transfer_time: float = 0.0
    t_ins: float = 0.0
    transfer_trajectory: Optional[np.ndarray] = None
    transfer_trajectory_times: Optional[np.ndarray] = None
    constraints_violation: float = 0.0


class Transfer:
    """DRO-RO 转移轨迹优化器

    提供基于 NLP 方法的 DRO（远距逆行轨道）到 RO（直线轨道）转移轨迹优化简化接口。

    Example:
        >>> from e2m2e.transfer import Transfer, TransferConfig
        >>> from e2m2e.core import CR3BP_System, CR3BP_Dynamics
        >>> from scripts.utils.common import MU
        >>>
        >>> system = CR3BP_System(mu=MU, primary="earth", secondary="moon")
        >>> dynamics = CR3BP_Dynamics(system=system)
        >>> transfer = Transfer(dynamics)
        >>> transfer.set_orbit(start=dro_orbit, end=ro_orbit)
        >>> result = transfer.optimize(
        ...     initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
        ...     alpha_range=(0.5, 2.5),
        ...     use_relaxed_velocity=True,
        ...     velocity_angle_tol=0.05,
        ... )
    """

    def __init__(self, dynamics: CR3BP_Dynamics):
        """初始化转移优化器

        Args:
            dynamics: CR3BP 动力学实例，用于轨道传播
        """
        self.dynamics = dynamics
        self.system = dynamics.system
        self.mu = self.system.mu

        self._departure_orbit: Optional[Orbit] = None
        self._arrival_orbit: Optional[Orbit] = None
        self._config = TransferConfig()
        self._result: Optional[TransferOptimizationResult] = None

    @property
    def departure_orbit(self) -> Optional[Orbit]:
        """出发轨道（DRO）。"""
        return self._departure_orbit

    @property
    def arrival_orbit(self) -> Optional[Orbit]:
        """到达轨道（RO）。"""
        return self._arrival_orbit

    @property
    def config(self) -> TransferConfig:
        """转移优化配置。"""
        return self._config

    @property
    def result(self) -> Optional[TransferOptimizationResult]:
        """最新优化结果。"""
        return self._result

    def set_orbit(self, start: Orbit, end: Orbit) -> "Transfer":
        """设置出发轨道和到达轨道

        Args:
            start: 出发轨道（DRO）
            end: 到达轨道（RO）

        Returns:
            self，支持链式调用
        """
        self._departure_orbit = start
        self._arrival_orbit = end
        return self

    def optimize(
        self,
        initial_guess: Dict[str, float],
        alpha_range: Tuple[float, float],
        departure_state: Optional[np.ndarray] = None,
        t_ins_range: Optional[Tuple[float, float]] = None,
        use_relaxed_velocity: Optional[bool] = None,
        velocity_angle_tol: Optional[float] = None,
    ) -> TransferOptimizationResult:
        """优化转移轨迹

        Args:
            initial_guess: 初始猜测，包含 'alpha'、'transfer_time'、't_ins'
            alpha_range: α 参数范围 (min, max)
            departure_state: 手动指定出发点状态 [6]；None 时自动从 DRO 采样
            t_ins_range: RO 上的插入时间范围，默认为完整 RO 周期
            use_relaxed_velocity: 覆盖配置中的 use_relaxed_velocity
            velocity_angle_tol: 覆盖配置中的 velocity_angle_tol

        Returns:
            TransferOptimizationResult，包含优化详情
        """
        if self._departure_orbit is None or self._arrival_orbit is None:
            raise ValueError("Must call set_orbit() before optimize()")

        # 若用户未指定覆盖值，使用配置中的默认值
        if use_relaxed_velocity is None:
            use_relaxed_velocity = self._config.use_relaxed_velocity
        if velocity_angle_tol is None:
            velocity_angle_tol = self._config.velocity_angle_tol

        # 出发点状态：用户手动指定或从 DRO 轨道采样
        if departure_state is None:
            departure_state = self._sample_departure_state_from_dro()
        else:
            departure_state = np.asarray(departure_state)

        # 插入时间范围：未指定时默认为一个完整 RO 周期
        if t_ins_range is None:
            t0 = self._arrival_orbit.times[0]
            period = self._get_ro_period()
            t_ins_range = (t0, t0 + period)

        # 构造 NLP 优化变量（α、转移时间、插入时间）
        ig = NLPOptimizationVariables(
            alpha=initial_guess["alpha"],
            transfer_time=initial_guess["transfer_time"],
            t_ins=initial_guess["t_ins"],
        )

        optimizer = DROTRONLPOptimizer(
            system=self.system,
            dynamics=self.dynamics,
            departure_orbit=self._departure_orbit,
            arrival_orbit=self._arrival_orbit,
            departure_state=departure_state,
        )

        optimizer.alpha_range = alpha_range
        optimizer.earth_radius = self._config.earth_radius
        optimizer.moon_radius = self._config.moon_radius
        optimizer.velocity_angle_tol = velocity_angle_tol
        optimizer.t_ins_range = t_ins_range

        if self._config.use_copt and _HAVE_COPT:
            nlp_result = optimize_with_copt(
                optimizer,
                initial_guess=ig,
                fallback_to_scipy=self._config.fallback_to_scipy,
            )
        else:
            nlp_result = optimizer.optimize(
                initial_guess=ig,
                alpha_range=alpha_range,
                t_ins_range=t_ins_range,
                use_relaxed_velocity_constraint=use_relaxed_velocity,
                velocity_angle_constraint=velocity_angle_tol,
                verbose=False,
            )

        self._result = self._convert_nlp_result(nlp_result, departure_state)
        return self._result

    def _sample_departure_state_from_dro(self) -> np.ndarray:
        """从 DRO 采样出发点状态，返回 DRO 轨道的第一个状态点。"""
        if self._departure_orbit is None:
            raise ValueError("Departure orbit not set")

        return self._departure_orbit.states[0].copy()

    def _get_ro_period(self) -> float:
        """获取 RO 轨道周期

        Returns:
            RO 周期；不可用时返回默认值 10.0
        """
        if self._arrival_orbit is None:
            return 10.0

        period = getattr(self._arrival_orbit, "period", None)
        if period is not None:
            return float(period)

        if hasattr(self._arrival_orbit, "times") and len(self._arrival_orbit.times) > 1:
            return float(self._arrival_orbit.times[-1] - self._arrival_orbit.times[0])

        return 10.0

    def _convert_nlp_result(
        self, nlp_result, departure_state: np.ndarray
    ) -> TransferOptimizationResult:
        """将 NLPOptimizationResult 转换为 TransferOptimizationResult

        Args:
            nlp_result: DROTRONLPOptimizer 的 NLP 优化结果
            departure_state: 实际使用的出发点状态

        Returns:
            TransferOptimizationResult
        """
        max_violation = 0.0
        if nlp_result.constraints_violation:
            max_violation = (
                max(nlp_result.constraints_violation.values())
                if nlp_result.constraints_violation
                else 0.0
            )

        return TransferOptimizationResult(
            success=nlp_result.success,
            message=nlp_result.message,
            departure_state=nlp_result.departure_state,
            departure_alpha=nlp_result.alpha,
            departure_beta=0.0,
            insertion_state=nlp_result.insertion_state,
            final_state=nlp_result.final_state,
            delta_v1=nlp_result.delta_v1,
            delta_v2=nlp_result.delta_v2,
            total_delta_v=nlp_result.objective_value,
            transfer_time=nlp_result.transfer_time,
            t_ins=nlp_result.t_ins,
            transfer_trajectory=nlp_result.transfer_trajectory,
            transfer_trajectory_times=nlp_result.transfer_times,
            constraints_violation=max_violation,
        )
