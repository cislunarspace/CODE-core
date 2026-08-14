"""CR3BP 相对运动动力学（主题 3）。

在目标轨道邻域内线性化 CR3BP 动力学，得到时变相对运动方程（RLM）。
相对状态 ρ = [δr, δv] 在会合系（Synodic）下定义，随目标轨道演化。

核心组件：
- :class:`TargetOrbit`：目标轨道包装，提供任意时刻状态查询
- :class:`RelativeDynamics`：RLM 线性化 + 相对状态/STM 传播

坐标系约定：相对状态默认在**会合系（Synodic）** 下定义（与 CR3BP 动力学
同系），LVLH 转换由调用方经 :class:`~e2m2e.algorithm.coordinate.LVLHAxes` 完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, overload

import numpy as np
import numpy.typing as npt

from e2m2e.integrators import RkMethod, solve_ivp_events

if TYPE_CHECKING:
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


class _SystemLike(Protocol):
    """动力学系统协议的最小接口（仅声明 `mu`）。"""

    mu: float


class DynamicsLike(Protocol):
    """鸭子类型协议：描述 ``CR3BP_Dynamics`` / ``EphemerisDynamics`` 的公共接口。

    相对运动动力学模块通过此协议访问底层绝对动力学对象，不依赖具体类。
    """

    system: _SystemLike

    @overload
    def compute_jacobian_A(self, state: npt.NDArray[np.floating]) -> np.ndarray: ...

    @overload
    def compute_jacobian_A(self, t: float, state: npt.NDArray[np.floating]) -> np.ndarray: ...

    def propagate(
        self,
        initial_state: npt.ArrayLike,
        t_span: tuple[float, float],
        t_eval: npt.ArrayLike | None = None,
        with_stm: bool = False,
        with_jacobi: bool = False,
        events: None = None,
    ) -> dict[str, Any]: ...

    def equations_of_motion(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]: ...


class RelativeDynamics:
    """相对运动动力学（RLM 时变线性化）。

    在目标轨道邻域内把动力学线性化，得到时变相对运动方程：

        δẋ = A(t) δx

    其中 A(t) 由 ``dynamics.compute_jacobian_A(t, state)`` 在目标状态处求值。
    支持 CR3BP（`CR3BP_Dynamics`）和星历（`EphemerisDynamics`）两套动力学，
    鸭子类型适配：只要对象有 ``compute_jacobian_A(t, state)`` 和
    ``propagate(..., with_stm=True)`` 即可。
    """

    def __init__(self, target: TargetOrbit, dynamics: DynamicsLike) -> None:
        self.target = target
        self.dynamics = dynamics

    def linear_model(self, t: float) -> np.ndarray:
        """返回 t 时刻 RLM 的 A(t) 矩阵，形状 ``(6, 6)``。

        CR3BP：A = [[0, I], [U, Ω]]，U 为伪势能 Hessian，Ω 为科里奥利项。
        星历：A = [[0, I], [∂a/∂r, 0]]，∂a/∂r 为 N 体引力雅可比。
        """
        state = self.target.state_at(t)
        # 鸭子类型适配：CR3BP 是 compute_jacobian_A(state)，
        # 星历是 compute_jacobian_A(t, state)
        try:
            return self.dynamics.compute_jacobian_A(t, state)
        except TypeError:
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

        在目标轨道上逐点线性化，用变步长 RK89 积分 δẋ = A(t)δx。

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

        # 均匀采样输出
        n = max(int((t_span[1] - t_span[0]) / (max_step or 0.01)) + 1, 2)
        t_eval = np.linspace(t_span[0], t_span[1], n)

        result = solve_ivp_events(
            t_span,
            rho0,
            t_eval,
            rtol,
            atol,
            eom,
            events=[],
            method=RkMethod.RK89,
            max_step=max_step,
        )
        times = np.asarray(result["time"], dtype=float)
        rhos = np.asarray(result["states"], dtype=float)
        # 防御性校验：Rust 侧提前退出（如 max_steps 耗尽）须暴露，不允许把
        # 截断结果当完整轨迹返回（issue #246，照抄 dynamics.py）。
        if len(times) != n:
            raise RuntimeError(f"相对传播返回 {len(times)} / {n} 个时间点，轨迹被截断")
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

    # ------------------------------------------------------------------
    # 非线性相对运动方程（牛顿式）与 Encke 改写
    # ------------------------------------------------------------------

    def _absolute_eom(self, state: np.ndarray) -> np.ndarray:
        """CR3BP 绝对动力学右端（会合系）。"""
        return self.dynamics.equations_of_motion(0.0, state)

    def nonlinear_eom(self, t: float, rho: np.ndarray) -> np.ndarray:
        """非线性相对运动方程右端（牛顿式，两式相减）。

        δẍ = f(x_target + δx) − f(x_target)

        近距离时两式相减产生截断误差（Cuevas del Valle 2022）。
        供对比验证用；实际传播建议用 :meth:`encke_eom`。

        Args:
            t: 时间
            rho: 相对状态 ``[δr, δv]``，形状 ``(6,)``

        Returns:
            相对状态导数 ``[δv, δa]``
        """
        target_state = self.target.state_at(t)
        chaser_state = target_state + rho
        f_target = self._absolute_eom(target_state)
        f_chaser = self._absolute_eom(chaser_state)
        return f_chaser - f_target

    def encke_eom(self, t: float, rho: np.ndarray) -> np.ndarray:
        """Encke 改写的非线性相对运动方程右端。

        把引力加速度差分项改写为 Encke 形式，避免近距离 ``1/r³``
        两式相减的截断误差。精度比牛顿式提升一个量级（Cuevas del Valle 2022）。

        对每个引力中心，Encke 公式（Battin 标准形式）：

            ``a_grav(r + δr) - a_grav(r) = -μ/|r|³ [f(q)·q·r + g(q)·δr]``

        其中 ``q = (2r·δr + |δr|²)/|r|²``，
        ``f(q) = ((1+q)^(−3/2) − 1)/q`` （q→0 时 f→−3/2，用级数展开），
        ``g(q) = (1+q)^(−3/2)``。

        Args:
            t: 时间
            rho: 相对状态 ``[δr, δv]``，形状 ``(6,)``

        Returns:
            相对状态导数 ``[δv, δa]``
        """
        mu = self.dynamics.system.mu
        target_state = self.target.state_at(t)
        dr = rho[:3]
        dv = rho[3:]

        # 伪势能梯度差（离心力 + 科里奥利力，线性项直接相减）
        x, y, z = target_state[0], target_state[1], target_state[2]
        dx, dy, _ = dr[0], dr[1], dr[2]

        # 离心力差（伪势能二次项）：∂Ω/∂x 中的 x 项
        da_centrifugal = np.array([dx, dy, 0.0])

        # 科里奥利力差：2ω×δv（CR3BP 旋转系角速度 ω = 1，z 轴）
        da_coriolis = np.array([2.0 * dv[1], -2.0 * dv[0], 0.0])

        # 引力差（Encke 改写）：两个引力中心
        # 主天体 1（质量 1−μ，位于 (−μ, 0, 0)）
        r1 = np.array([x + mu, y, z])
        r1_norm = np.linalg.norm(r1)
        q1 = (2.0 * np.dot(r1, dr) + np.dot(dr, dr)) / (r1_norm * r1_norm)
        f1 = _encke_f(q1)
        g1 = (1.0 + q1) ** (-1.5)
        da_grav1 = -(1.0 - mu) / (r1_norm**3) * (f1 * q1 * r1 + g1 * dr)

        # 主天体 2（质量 μ，位于 (1−μ, 0, 0)）
        r2 = np.array([x - 1.0 + mu, y, z])
        r2_norm = np.linalg.norm(r2)
        q2 = (2.0 * np.dot(r2, dr) + np.dot(dr, dr)) / (r2_norm * r2_norm)
        f2 = _encke_f(q2)
        g2 = (1.0 + q2) ** (-1.5)
        da_grav2 = -mu / (r2_norm**3) * (f2 * q2 * r2 + g2 * dr)

        da = da_centrifugal + da_coriolis + da_grav1 + da_grav2
        return np.concatenate([dv, da])

    def propagate_nonlinear(
        self,
        rho0: npt.ArrayLike,
        t_span: tuple[float, float],
        *,
        method: str = "encke",
        rtol: float = 1e-10,
        atol: float = 1e-12,
        max_step: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """传播相对状态（非线性方程）。

        Args:
            rho0: 初始相对状态 ``[δr, δv]``，形状 ``(6,)``
            t_span: 传播区间
            method: ``"encke"`` （推荐）或 ``"newton"`` （两式相减，精度低）
            rtol, atol: 积分容差
            max_step: 最大步长

        Returns:
            ``(times, rhos)``
        """
        rho0 = np.asarray(rho0, dtype=float)
        if rho0.shape != (6,):
            raise ValueError(f"rho0 形状须为 (6,)，得到 {rho0.shape}")

        eom = self.encke_eom if method == "encke" else self.nonlinear_eom
        n = max(int((t_span[1] - t_span[0]) / (max_step or 0.01)) + 1, 2)
        t_eval = np.linspace(t_span[0], t_span[1], n)

        result = solve_ivp_events(
            t_span,
            rho0,
            t_eval,
            rtol,
            atol,
            eom,
            events=[],
            method=RkMethod.RK89,
            max_step=max_step,
        )
        times = np.asarray(result["time"], dtype=float)
        rhos = np.asarray(result["states"], dtype=float)
        if len(times) != n:
            raise RuntimeError(f"非线性相对传播返回 {len(times)} / {n} 个时间点，轨迹被截断")
        return times, rhos

    # ------------------------------------------------------------------
    # 星历相对运动（EphemerisDynamics）
    # ------------------------------------------------------------------

    def ephemeris_linear_model(self, t: float) -> np.ndarray:
        """星历 RLM 的 A(t) 矩阵（同 :meth:`linear_model`，显式命名）。"""
        return self.linear_model(t)

    def ephemeris_encke_eom(self, t: float, rho: np.ndarray) -> np.ndarray:
        """星历 Encke 改写的非线性相对运动方程右端（预留）。

        星历惯性系无科里奥利/离心项，Encke 公式需按 N 体逐项改写
        （每个天体的直接项差 Encke 化，间接项与位置无关直接相消）。
        当前版本暂用牛顿式两式相减，后续实现。

        Args:
            t: 时间
            rho: 相对状态 ``[δr, δv]``，形状 ``(6,)``

        Returns:
            相对状态导数 ``[δv, δa]``
        """
        # 当前用牛顿式；Encke 星历版留后续（需访问 ForceModel 各力分量）
        return self.nonlinear_eom(t, rho)

    # ------------------------------------------------------------------
    # LVLH 系相对状态转换
    # ------------------------------------------------------------------

    def to_lvlh(self, rho_syn: npt.ArrayLike, t: float) -> tuple[np.ndarray, np.ndarray]:
        """把会合系相对状态转换到 LVLH 系。

        LVLH 系定义（`LVLHAxes`）：R = 径向，H = 角动量，V = H × R。
        转换公式：

            ρ_lvlh = Rᵀ δr_syn
            ρ̇_lvlh = Rᵀ δv_syn + Ṙᵀ δr_syn

        其中 R 为目标轨道的 LVLH 旋转矩阵，Ṙ 用中心差分近似
        （步长 1e-5，与 `Axes.rotation_and_rate` 一致）。

        Args:
            rho_syn: 会合系相对状态 ``[δr, δv]``，形状 ``(6,)``
            t: 参考历元

        Returns:
            ``(rho_lvlh, rho_dot_lvlh)``：LVLH 系相对位置与速度，
            各形状 ``(3,)``
        """
        rho_syn = np.asarray(rho_syn, dtype=float)
        dr_syn = rho_syn[:3]
        dv_syn = rho_syn[3:]

        # 目标轨道在 t 时刻的状态
        target_state = self.target.state_at(t)
        r_target = target_state[:3]
        v_target = target_state[3:]

        # LVLH 旋转矩阵（目标轨道）
        R = _lvlh_rotation(r_target, v_target)

        # R 的时间导数（中心差分）
        dt = 1e-5
        state_plus = self.target.state_at(t + dt)
        state_minus = self.target.state_at(t - dt)
        R_plus = _lvlh_rotation(state_plus[:3], state_plus[3:])
        R_minus = _lvlh_rotation(state_minus[:3], state_minus[3:])
        R_dot = (R_plus - R_minus) / (2.0 * dt)

        # 位置转换
        rho_lvlh = R.T @ dr_syn

        # 速度转换：ρ̇_lvlh = Rᵀ δv_syn + Ṙᵀ δr_syn
        rho_dot_lvlh = R.T @ dv_syn + R_dot.T @ dr_syn

        return rho_lvlh, rho_dot_lvlh

    def from_lvlh(
        self, rho_lvlh: npt.ArrayLike, rho_dot_lvlh: npt.ArrayLike, t: float
    ) -> np.ndarray:
        """把 LVLH 系相对状态转换回会合系。

        逆转换：

            δr_syn = R ρ_lvlh
            δv_syn = R ρ̇_lvlh + Ṙ ρ_lvlh

        Args:
            rho_lvlh: LVLH 系相对位置，形状 ``(3,)``
            rho_dot_lvlh: LVLH 系相对速度，形状 ``(3,)``
            t: 参考历元

        Returns:
            会合系相对状态 ``[δr, δv]``，形状 ``(6,)``
        """
        rho_lvlh = np.asarray(rho_lvlh, dtype=float)
        rho_dot_lvlh = np.asarray(rho_dot_lvlh, dtype=float)

        target_state = self.target.state_at(t)
        r_target = target_state[:3]
        v_target = target_state[3:]
        R = _lvlh_rotation(r_target, v_target)

        dt = 1e-5
        state_plus = self.target.state_at(t + dt)
        state_minus = self.target.state_at(t - dt)
        R_plus = _lvlh_rotation(state_plus[:3], state_plus[3:])
        R_minus = _lvlh_rotation(state_minus[:3], state_minus[3:])
        R_dot = (R_plus - R_minus) / (2.0 * dt)

        dr_syn = R @ rho_lvlh
        dv_syn = R @ rho_dot_lvlh + R_dot @ rho_lvlh

        return np.concatenate([dr_syn, dv_syn])


def _lvlh_rotation(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """构造 LVLH 旋转矩阵 R = [r_hat, v_hat, h_hat]。

    R = 径向，H = 角动量方向，V = H × R。
    返回 (3, 3)，列为单位向量。
    """
    h = np.cross(r, v)
    r_hat = r / np.linalg.norm(r)
    h_hat = h / np.linalg.norm(h)
    v_hat = np.cross(h_hat, r_hat)
    return np.column_stack([r_hat, v_hat, h_hat])


def _encke_f(q: float) -> float:
    """Encke 函数 f(q) = ((1+q)^(−3/2) − 1) / q。

    q→0 时极限为 −3/2。用泰勒展开避免 0/0：
    f(q) ≈ −3/2 + 15/8 q − 35/16 q² + ...
    """
    if abs(q) < 1e-6:
        # 泰勒展开（q 很小时）
        return -1.5 + 1.875 * q - 2.1875 * q * q
    return ((1.0 + q) ** (-1.5) - 1.0) / q
