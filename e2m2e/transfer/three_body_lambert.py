"""三体 Lambert 打靶模块。

以二体 Lambert 解为初猜，在 CR3BP 动力学下用 Newton 打靶修正出发速度，
使给定飞行时间后的末端位置命中目标。流程（孙俞等 2017、Fossa 等 2022）：

1. 两端物理状态 (km, km/s) 经 ``CR3BP_System.physical_to_dimensionless`` 无量纲化；
2. 无量纲几何上以 μ = 1 调 :func:`solve_lambert` 得初猜出发速度；
3. Newton 迭代：传播 ``with_stm=True``，取末端 STM 的 Φ_rv 块（``Φ[0:3, 3:6]``）
   解修正量，收敛判据为末端位置误差 < 1e-8（无量纲）。
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from ..core.dynamics import CR3BP_Dynamics
from .config import TransferArc, TransferSolution
from .lambert import solve_lambert
from .terminal import StateTerminal

logger = logging.getLogger(__name__)


class ThreeBodyLambert:
    """CR3BP 两点边值打靶求解器。

    Attributes:
        dynamics: CR3BP_Dynamics 对象（system 须已初始化特征尺度）
        tolerance: 末端位置误差收敛判据（无量纲）
        max_iterations: 最大 Newton 迭代次数
    """

    DEFAULT_TOLERANCE = 1e-8
    DEFAULT_MAX_ITERATIONS = 20
    # 发散保护：误差超过该值（无量纲长度，约 10 倍地月距离量级）即判失败
    DIVERGENCE_LIMIT = 10.0

    def __init__(self, dynamics: CR3BP_Dynamics) -> None:
        """初始化

        Args:
            dynamics: CR3BP_Dynamics 对象

        Raises:
            ValueError: 系统未初始化特征尺度（物理/无量纲换算不可用）
        """
        system = getattr(dynamics, "system", None)
        if system is None or not getattr(system, "is_initialized", False):
            raise ValueError("dynamics 的 system 须已初始化特征尺度")
        self.dynamics = dynamics
        self.system = system
        self.tolerance = self.DEFAULT_TOLERANCE
        self.max_iterations = self.DEFAULT_MAX_ITERATIONS

    def solve(
        self,
        term0: StateTerminal,
        term1: StateTerminal,
        tof: float,
        guess: Literal["lambert", "orbit"] = "lambert",
    ) -> TransferSolution:
        """解三体 Lambert 问题：修正出发速度使 tof 后末端位置命中 term1。

        Args:
            term0: 出发终端，状态为物理单位 (km, km/s)
            term1: 到达终端，状态为物理单位 (km, km/s)；仅位置为约束，
                速度用于计算到达脉冲
            tof: 飞行时间，s
            guess: 初猜来源；``"lambert"`` 用无量纲二体 Lambert 解，
                ``"orbit"`` 直接用 term0 的速度

        Returns:
            :class:`TransferSolution`，单弧，物理单位；未收敛时 ``converged=False``
        """
        x0 = self.system.physical_to_dimensionless(term0.state)
        x1 = self.system.physical_to_dimensionless(term1.state)
        tof_dim = float(tof) / self.system.characteristic_time
        if tof_dim <= 0:
            raise ValueError(f"tof 必须为正数，当前为 {tof}")

        r0, rf = x0[:3], x1[:3]
        if guess == "lambert":
            v0 = solve_lambert(r0, rf, tof_dim, mu=1.0).v0
        elif guess == "orbit":
            v0 = np.array(x0[3:], copy=True)
        else:
            raise ValueError(f"guess 必须是 'lambert' 或 'orbit'，得到 {guess!r}")

        v0, error, n_iter, converged = self._shoot(r0, rf, v0, tof_dim)

        # 末次传播取收敛弧（未收敛时取最后一次迭代弧）
        result = self.dynamics.propagate(
            np.concatenate([r0, v0]), (0.0, tof_dim), t_eval=np.linspace(0.0, tof_dim, 200)
        )
        states_phys = np.array([self.system.dimensionless_to_physical(s) for s in result["states"]])
        times_phys = np.asarray(result["time"], dtype=float) * self.system.characteristic_time

        v_char = self.system.characteristic_velocity
        delta_v1 = float(np.linalg.norm(v0 - x0[3:]) * v_char)
        arrival_delta_v = float(np.linalg.norm(result["states"][-1][3:] - x1[3:]) * v_char)

        message = "" if converged else f"Newton 未收敛：末端位置误差 {error:.3e}（无量纲）"
        return TransferSolution(
            arcs=(TransferArc(states=states_phys, times=times_phys, delta_v=delta_v1),),
            arrival_delta_v=arrival_delta_v,
            total_delta_v=delta_v1 + arrival_delta_v,
            transfer_time=float(tof),
            converged=converged,
            n_iter=n_iter,
            message=message,
        )

    # ---- 内部实现 ----

    # 线搜索最大回退次数（阻尼 Newton：整步增大误差时步长减半）
    _LINE_SEARCH_STEPS = 10

    def _shoot(
        self, r0: np.ndarray, rf: np.ndarray, v0: np.ndarray, tof_dim: float
    ) -> tuple[np.ndarray, float, int, bool]:
        """阻尼 Newton 打靶：修正 v0 使传播 tof_dim 后位置命中 rf。

        雅各比取末端 STM 的 Φ_rv 块；奇异时回退最小二乘；整步使误差增大时
        步长减半（初猜离解较远时整步 Newton 易跑出收敛域）。
        发散保护：误差非有限、超过 DIVERGENCE_LIMIT 或线搜索失败时中止。
        """
        v0 = np.array(v0, dtype=float)
        error = np.inf
        for n_iter in range(1, self.max_iterations + 1):
            result = self.dynamics.propagate(
                np.concatenate([r0, v0]), (0.0, tof_dim), with_stm=True
            )
            err = result["states"][-1][:3] - rf
            error = float(np.linalg.norm(err))
            if not np.isfinite(error) or error > self.DIVERGENCE_LIMIT:
                logger.warning("三体打靶发散：迭代 %d 误差 %.3e", n_iter, error)
                return v0, error, n_iter, False
            if error < self.tolerance:
                return v0, error, n_iter, True

            phi_rv = np.asarray(result["stm"])[-1][0:3, 3:6]
            try:
                delta = np.linalg.solve(phi_rv, err)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(phi_rv, err, rcond=None)[0]

            alpha = 1.0
            for _ in range(self._LINE_SEARCH_STEPS):
                trial = v0 - alpha * delta
                result_trial = self.dynamics.propagate(np.concatenate([r0, trial]), (0.0, tof_dim))
                error_trial = float(np.linalg.norm(result_trial["states"][-1][:3] - rf))
                if np.isfinite(error_trial) and error_trial < error:
                    v0 = trial
                    break
                alpha *= 0.5
            else:
                logger.warning("三体打靶线搜索失败：迭代 %d 误差 %.3e", n_iter, error)
                return v0, error, n_iter, False

        return v0, error, self.max_iterations, False
