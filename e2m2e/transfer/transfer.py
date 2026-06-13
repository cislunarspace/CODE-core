"""DRO-RO 转移轨迹优化模块

提供基于 NLP 方法（Cui et al. 2025）的简化转移轨迹优化接口。
"""

from __future__ import annotations

import numpy as np

from ..core.dynamics import CR3BP_Dynamics
from ..core.orbit import Orbit
from . import transfer_optimization
from .config import TransferConfig, TransferOptimizationResult
from .optimizers import COPTTransferOptimizer, SciPyTransferOptimizer
from .transfer_optimization import DROTRONLPOptimizer, NLPOptimizationVariables

_HAVE_COPT = transfer_optimization.coptpy is not None


DU = 3.84405000e5


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

        self._departure_orbit: Orbit | None = None
        self._arrival_orbit: Orbit | None = None
        self._config = TransferConfig()
        self._result: TransferOptimizationResult | None = None

    @property
    def departure_orbit(self) -> Orbit | None:
        """出发轨道（DRO）。"""
        return self._departure_orbit

    @property
    def arrival_orbit(self) -> Orbit | None:
        """到达轨道（RO）。"""
        return self._arrival_orbit

    @property
    def config(self) -> TransferConfig:
        """转移优化配置。"""
        return self._config

    @property
    def result(self) -> TransferOptimizationResult | None:
        """最新优化结果。"""
        return self._result

    def set_orbit(self, start: Orbit, end: Orbit) -> Transfer:
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
        initial_guess: dict[str, float],
        alpha_range: tuple[float, float],
        departure_state: np.ndarray | None = None,
        t_ins_range: tuple[float, float] | None = None,
        use_relaxed_velocity: bool | None = None,
        velocity_angle_tol: float | None = None,
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

        # 通过 config 一次性传入所有优化参数，避免 poke optimizer 属性
        config = TransferConfig(
            alpha_min=alpha_range[0],
            alpha_max=alpha_range[1],
            t_ins_range=t_ins_range,
            earth_radius=self._config.earth_radius,
            moon_radius=self._config.moon_radius,
            use_relaxed_velocity=use_relaxed_velocity,
            velocity_angle_tol=velocity_angle_tol,
            use_copt=self._config.use_copt,
            fallback_to_scipy=self._config.fallback_to_scipy,
            verbose=False,
        )

        optimizer = DROTRONLPOptimizer(
            system=self.system,
            dynamics=self.dynamics,
            departure_orbit=self._departure_orbit,
            arrival_orbit=self._arrival_orbit,
            departure_state=departure_state,
            config=config,
        )

        adapter = self._build_optimizer_adapter(optimizer)
        nlp_result = adapter.optimize(initial_guess=ig)

        self._result = nlp_result
        return self._result

    def _build_optimizer_adapter(self, optimizer: DROTRONLPOptimizer):
        """根据配置构造 SciPy 或 COPT adapter。"""
        if self._config.use_copt and _HAVE_COPT:
            return COPTTransferOptimizer(
                optimizer,
                fallback_to_scipy=self._config.fallback_to_scipy,
            )
        return SciPyTransferOptimizer(optimizer)

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
