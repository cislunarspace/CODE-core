"""调相设计：同轨道不同相位间转移（主题 3）。

基于 STM 的两脉冲调相（Fossa 2022 NRHO 范式）：在目标周期轨道上，
从相位 φ₀ 到相位 φ₁ = φ₀ + Δφ，用两脉冲完成转移。

算法：对每个飞行时间 tof，用相对 STM 解两点边值：
    δr_f = Φ_rr δr₀ + Φ_rv (δv₀ + Δv₁)
    0    = Φ_vr δr₀ + Φ_vv (δv₀ + Δv₁) + Δv₂

其中 δr₀/δv₀ 是初始相对状态（由相位差决定），δr_f = 0（到达目标相位），
Δv₁/Δv₂ 是待求脉冲。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ...data.templates import ConvergenceState, FailureCause
from ..proximity.relative_dynamics import DynamicsLike, RelativeDynamics, TargetOrbit
from ..results import ResultStatus

if TYPE_CHECKING:
    from ...data.types.orbit import Orbit


@dataclass
class PhasingManeuver:
    """调相脉冲。

    Attributes:
        t: 脉冲时刻（与目标轨道时间坐标一致）
        dv: 脉冲速度增量，形状 ``(3,)``，无量纲（CR3BP DU/TU）
    """

    t: float
    dv: np.ndarray


@dataclass
class PhasingSolution:
    """调相解。

    Attributes:
        maneuvers: 脉冲序列（通常 2 个）
        tof: 飞行时间
        total_dv: 总脉冲大小（标量）
        converged: 是否收敛
    """

    maneuvers: list[PhasingManeuver]
    tof: float
    total_dv: float
    status: ConvergenceState
    cause: FailureCause
    message: str

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


def phasing_search(
    orbit: Orbit,
    dphase: float,
    tof_grid: npt.ArrayLike,
    dynamics: DynamicsLike,
    *,
    t0: float | None = None,
) -> list[PhasingSolution]:
    """调相搜索：在 tof 网格上求两脉冲调相解。

    对周期轨道，相位差 dphase 对应目标轨道上的时间差
    ``dt = dphase / (2π) * period``。初始相对状态由目标轨道在
    ``t0`` 与 ``t0 + dt`` 的状态差给出。

    对每个 tof，用相对 STM 解两点边值求两脉冲。

    Args:
        orbit: 目标周期轨道
        dphase: 相位差（rad），``2π`` 为一整圈
        tof_grid: 飞行时间网格，形状 ``(m,)``，与轨道时间坐标一致
        dynamics: CR3BP 动力学
        t0: 出发时刻，None 时取轨道起点

    Returns:
        每个 tof 对应的 :class:`PhasingSolution` 列表
    """
    target = TargetOrbit(orbit)
    rd = RelativeDynamics(target, dynamics)
    tof_grid = np.atleast_1d(np.asarray(tof_grid, dtype=float))

    if t0 is None:
        t0 = float(orbit.times[0])
    period = float(orbit.times[-1] - orbit.times[0])

    # 相位差对应的时间差
    dt_phase = dphase / (2.0 * np.pi) * period

    # 初始相对状态：目标相位 vs 出发相位
    state_dep = target.state_at(t0)
    state_arr = target.state_at(t0 + dt_phase)
    rho0 = state_arr - state_dep  # [δr, δv]

    solutions = []
    for tof in tof_grid:
        tof = float(tof)
        if tof <= 0:
            continue
        sol = _solve_two_impulse(rd, rho0, t0, tof)
        solutions.append(sol)
    return solutions


def _solve_two_impulse(
    rd: RelativeDynamics,
    rho0: np.ndarray,
    t0: float,
    tof: float,
) -> PhasingSolution:
    """基于 STM 的两脉冲调相求解。

    方程：
        δr_f = Φ_rr δr₀ + Φ_rv (δv₀ + Δv₁) = 0
        解出 Δv₁ = -Φ_rv⁻¹ (Φ_rr δr₀ + Φ_rv δv₀)
               = -Φ_rv⁻¹ Φ_rr δr₀ - δv₀

    第二脉冲在末端消除剩余速度：
        Δv₂ = -Φ_vr δr₀ - Φ_vv (δv₀ + Δv₁)
    """
    t_span = (t0, t0 + tof)
    _, _, stms = rd.propagate_with_stm(rho0, t_span)
    Phi = stms[-1]  # (6, 6)

    Phi_rr = Phi[:3, :3]
    Phi_rv = Phi[:3, 3:]
    Phi_vr = Phi[3:, :3]
    Phi_vv = Phi[3:, 3:]

    dr0 = rho0[:3]
    dv0 = rho0[3:]

    # 第一脉冲：使末端位置为零
    # Φ_rr δr₀ + Φ_rv (δv₀ + Δv₁) = 0
    # Δv₁ = -Φ_rv⁻¹ (Φ_rr δr₀ + Φ_rv δv₀)
    try:
        Phi_rv_inv = np.linalg.inv(Phi_rv)
        dv1 = -Phi_rv_inv @ (Phi_rr @ dr0 + Phi_rv @ dv0)
    except np.linalg.LinAlgError:
        return PhasingSolution(
            maneuvers=[],
            tof=tof,
            total_dv=np.inf,
            status=ConvergenceState.FAILED,
            cause=FailureCause.SINGULAR_JACOBIAN,
            message="调相状态转移矩阵奇异",
        )

    # 第二脉冲：消除末端速度
    # δv_f = Φ_vr δr₀ + Φ_vv (δv₀ + Δv₁)
    # Δv₂ = -δv_f
    dv_f = Phi_vr @ dr0 + Phi_vv @ (dv0 + dv1)
    dv2 = -dv_f

    maneuvers = [
        PhasingManeuver(t=t0, dv=dv1),
        PhasingManeuver(t=t0 + tof, dv=dv2),
    ]
    total_dv = float(np.linalg.norm(dv1) + np.linalg.norm(dv2))
    return PhasingSolution(
        maneuvers=maneuvers,
        tof=tof,
        total_dv=total_dv,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="调相求解完成",
    )
