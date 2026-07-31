"""CR3BP 相对运动动力学（主题 3）。

在目标轨道邻域内线性化 CR3BP 动力学，得到时变相对运动方程（RLM）。
相对状态 ρ = [δr, δv] 在会合系（Synodic）下定义，随目标轨道演化。

核心组件：
- :class:`TargetOrbit`：目标轨道包装，提供任意时刻状态查询
- :class:`RelativeDynamics`：RLM 线性化 + 相对状态/STM 传播

坐标系约定：相对状态默认在**会合系（Synodic）**下定义（与 CR3BP 动力学
同系），LVLH 转换由调用方经 :class:`~e2m2e.core.coordinate.LVLHAxes` 完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from ..core.dynamics import CR3BP_Dynamics
    from ..core.orbit import Orbit


@dataclass
class RelativeState:
    """相对状态（会合系）。

    Attributes:
        rho: 相对位置，形状 ``(3,)``，无量纲（CR3BP DU）
        rho_dot: 相对速度，形状 ``(3,)``，无量纲（CR3BP DU/TU）
        frame: 坐标系标签，``"Synodic"`` 或 ``"LVLH"``
        epoch: 参考历元（目标轨道时间坐标）
    """

    rho: np.ndarray
    rho_dot: np.ndarray
    frame: str
    epoch: float


class TargetOrbit:
    """目标轨道包装：任意时刻状态查询。

    线性插值（C⁰ 连续）。对长期高精度需求，后续可升级为三次样条。
    """

    def __init__(self, orbit: Orbit) -> None:
        self._orbit = orbit
        self._times = np.asarray(orbit.times, dtype=float)
        self._states = np.asarray(orbit.states, dtype=float)
        if self._times.shape[0] < 2:
            raise ValueError("目标轨道至少需要 2 个数据点")

    @property
    def t_span(self) -> tuple[float, float]:
        """轨道覆盖的时间范围。"""
        return float(self._times[0]), float(self._times[-1])

    def state_at(self, t: float) -> np.ndarray:
        """线性插值取 t 时刻目标状态，形状 ``(6,)``。

        Args:
            t: 时间（与轨道 times 同坐标）

        Returns:
            目标状态 ``[x, y, z, vx, vy, vz]``

        Raises:
            ValueError: t 超出轨道时间范围
        """
        t = float(t)
        t0, t1 = self.t_span
        if t < t0 or t > t1:
            raise ValueError(f"查询时间 {t} 超出轨道范围 [{t0}, {t1}]")
        idx = int(np.searchsorted(self._times, t, side="right")) - 1
        idx = min(idx, self._times.shape[0] - 2)
        dt = self._times[idx + 1] - self._times[idx]
        w = (t - self._times[idx]) / dt if dt > 0 else 0.0
        return (1 - w) * self._states[idx] + w * self._states[idx + 1]


class RelativeDynamics:
    """CR3BP 相对运动动力学（RLM 时变线性化）。

    在目标轨道邻域内把 CR3BP 动力学线性化，得到时变相对运动方程：

        δẋ = A(t) δx

    其中 A(t) = :meth:`CR3BP_Dynamics.compute_jacobian_A` 在目标状态处求值。
    """

    def __init__(self, target: TargetOrbit, dynamics: CR3BP_Dynamics) -> None:
        self.target = target
        self.dynamics = dynamics

    def linear_model(self, t: float) -> np.ndarray:
        """返回 t 时刻 RLM 的 A(t) 矩阵，形状 ``(6, 6)``。

        A = [[0, I], [U, Ω]]，U 为伪势能 Hessian 在目标状态处求值，
        Ω 为科里奥利项。
        """
        state = self.target.state_at(t)
        return self.dynamics.compute_jacobian_A(state)

    def propagate(
        self,
        rho0: npt.ArrayLike,
        t_span: tuple[float, float],
        *,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        max_step: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """传播相对状态（6 维）。

        在目标轨道上逐点线性化，用变步长 RK45 积分 δẋ = A(t)δx。

        Args:
            rho0: 初始相对状态 ``[δr, δv]``，形状 ``(6,)``
            t_span: 传播区间 ``(t0, tf)``
            rtol, atol: 积分容差
            max_step: 最大步长，None 时自动

        Returns:
            ``(times, rhos)``：times 形状 ``(n,)``，rhos 形状 ``(n, 6)``
        """
        rho0 = np.asarray(rho0, dtype=float)
        if rho0.shape != (6,):
            raise ValueError(f"rho0 形状须为 (6,)，得到 {rho0.shape}")

        def eom(t: float, rho: np.ndarray) -> np.ndarray:
            A = self.linear_model(t)
            return A @ rho

        result = solve_ivp(
            eom,
            t_span,
            rho0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            max_step=max_step if max_step is not None else np.inf,
            dense_output=True,
        )
        if not result.success:
            raise RuntimeError(f"相对传播失败: {result.message}")

        # 均匀采样输出
        n = max(int((t_span[1] - t_span[0]) / (max_step or 0.01)) + 1, 2)
        times = np.linspace(t_span[0], t_span[1], n)
        rhos = result.sol(times).T
        return times, rhos

    def propagate_with_stm(
        self,
        rho0: npt.ArrayLike,
        t_span: tuple[float, float],
        *,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """传播相对状态 + 相对 STM（42 维增广）。

        相对 STM Φ_rel(t, t₀) 满足 δx(t) = Φ_rel(t, t₀) δx(t₀)。
        由于线性化在同一会合系下，Φ_rel = Φ_abs（绝对 STM）。
        本方法复用 :meth:`CR3BP_Dynamics.propagate` 的 with_stm=True，
        在目标轨道上传播绝对 STM，即得相对 STM。

        Args:
            rho0: 初始相对状态 ``[δr, δv]``，形状 ``(6,)``
            t_span: 传播区间

        Returns:
            ``(times, rhos, stms)``：stms 形状 ``(n, 6, 6)``
        """
        rho0 = np.asarray(rho0, dtype=float)
        # 用目标轨道初始状态传播绝对 STM
        state0 = self.target.state_at(t_span[0])
        result = self.dynamics.propagate(
            state0, t_span, with_stm=True, t_eval=np.linspace(*t_span, 100)
        )
        stms = result["stm"]  # (n, 6, 6)
        times = result["time"]

        # 相对状态用 STM 直接计算（比线性化积分更一致）
        # δx(t) = Φ(t, t₀) δx(t₀)
        rhos = np.empty((times.shape[0], 6))
        for i in range(times.shape[0]):
            rhos[i] = stms[i] @ rho0

        return times, rhos, stms
