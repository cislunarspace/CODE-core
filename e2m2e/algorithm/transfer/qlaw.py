"""Q-law 低推力初猜生成器（Lyapunov 反馈律）。

用 Q-law（Petropoulos；Holt 2024 式 6-10）做低推力转移的初猜生成：前向反馈
积分产出次优控制历史，喂 :class:`~e2m2e.transfer.lowthrust_shooting.LowThrustShooting`
做解析雅可比打磨。是 gap-analysis「Q-law 作初猜 → 打靶优化」两级流程的上半段。

## 最简版（控 a, e, i）

只控半长轴、偏心率、倾角，规避 ω̇/Ω̇ 的 1/e、1/sin i 奇异。Q 函数：

```
Q = Σ_{X∈{a,e,i}} (δ(X, X_T) / max_ν(Ẋ))²
```

控制律（Holt 式 8，最速下降）：``u_RTN = −f·BᵀMᵀ/||MB||``，其中 f=T/m、
B 是 Gauss 方程 3×3 子阵、M=∂Q/∂[a,e,i]（中心差分）。油门固定满推。

## 架构：分段常量控制 + Rust 接龙

Q-law 不用连续方向回调（现有传播只接受固定方向），而是把 ``[t0,tf]`` 分成 N
段，每段用 Q-law 在段初算一个固定 ``(throttle, θ₁, θ₂)``，调
:func:`propagate_compiled_lowthrust`（复用地基 Rust 7D 受控传播）传播该段，
段末状态作下段初态。产出直接是 :class:`LowThrustSegment` 序列，与求解器决策
变量对齐。详见 ``docs/plans/qlaw-prd.md``。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ...data.constants import Datum
from ..forces import PhysicalModel
from .lowthrust_shooting import EngineConfig, LowThrustSegment

if TYPE_CHECKING:
    from ..dynamics import System
    from .lowthrust_shooting import LowThrustShooting

_G0 = 9.81  # m/s²，与 lowthrust_shooting 一致


def rv_to_keplerian(
    r: npt.NDArray[np.floating], v: npt.NDArray[np.floating], mu: float
) -> tuple[float, float, float, float, float, float]:
    """笛卡尔状态 → 经典开普勒根数 ``(a, e, i, Ω, ω, ν)``。

    标准算法（Bate-Müller-White）。角度单位弧度，a 单位同 r（km）。
    圆轨道（e≈0）ω/ν 单独无意义，但 ω+ν（近地点幅角+真近点角=纬度幅角）稳定。
    """
    r_vec = np.asarray(r, dtype=float)
    v_vec = np.asarray(v, dtype=float)
    r_norm = float(np.linalg.norm(r_vec))
    v_norm = float(np.linalg.norm(v_vec))

    # 能量 → 半长轴
    energy = v_norm**2 / 2.0 - mu / r_norm
    a = -mu / (2.0 * energy)

    # 角动量矢量 → 倾角、升交点赤经
    h_vec = np.cross(r_vec, v_vec)
    h_norm = float(np.linalg.norm(h_vec))
    n_vec = np.array([-h_vec[1], h_vec[0], 0.0])  # 升交点矢量 = k × h
    n_norm = float(np.linalg.norm(n_vec))

    i = float(np.arccos(np.clip(h_vec[2] / h_norm, -1.0, 1.0)))

    # 偏心率矢量 → 偏心率、近地点幅角
    e_vec = ((v_norm**2 - mu / r_norm) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = float(np.linalg.norm(e_vec))

    # 升交点赤经 Ω（赤道轨道 n_norm≈0 时未定义，给 0）
    if n_norm > 1e-12:
        omega_raan = float(np.arccos(np.clip(n_vec[0] / n_norm, -1.0, 1.0)))
        if n_vec[1] < 0:
            omega_raan = 2 * np.pi - omega_raan
    else:
        omega_raan = 0.0

    # 近地点幅角 ω（圆轨道 e≈0 时未定义，给 0）
    if e > 1e-10 and n_norm > 1e-12:
        argp = float(np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_norm * e), -1.0, 1.0)))
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp
    elif e > 1e-10:
        argp = float(np.arccos(np.clip(e_vec[0] / e, -1.0, 1.0)))
        if e_vec[1] < 0:
            argp = 2 * np.pi - argp
    else:
        argp = 0.0

    # 真近点角 ν
    if e > 1e-10:
        nu = float(np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r_norm), -1.0, 1.0)))
        if np.dot(r_vec, v_vec) < 0:
            nu = 2 * np.pi - nu
    else:
        # 圆轨道：用纬度幅角 u = ω + ν 的余弦近似 ν
        if n_norm > 1e-12:
            u = float(np.arccos(np.clip(np.dot(n_vec, r_vec) / (n_norm * r_norm), -1.0, 1.0)))
            if r_vec[2] < 0:
                u = 2 * np.pi - u
        else:
            u = float(np.arccos(np.clip(r_vec[0] / r_norm, -1.0, 1.0)))
            if r_vec[1] < 0:
                u = 2 * np.pi - u
        nu = u  # 圆轨道 ω=0，ν≈纬度幅角

    return a, e, i, omega_raan, argp, nu


def _max_rates_aei(
    a: float, e: float, i: float, omega: float, f: float, mu: float
) -> tuple[float, float, float]:
    """各根数 max_ν(Ẋ)（对真近点角 ν 网格扫描最大值），f=T_max/m（km/s²）。

    返回 (ȧ_max, ė_max, ị_max)。f=0 时全返 1.0（避免除零，Q 不再驱动）。

    用 ν 网格扫描而非闭式：闭式（Petropoulos）在圆/赤道轨道易出 0/0，扫描
    稳健。B 矩阵各元素对 ν 的最大值即 max_ν(Ẋ)（对推力方向最优后）。
    """
    if f <= 0.0:
        return 1.0, 1.0, 1.0
    p = a * (1.0 - e * e)
    h = float(np.sqrt(mu * p))
    n_grid = 18
    nus = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    a_max = e_max = i_max = 0.0
    for nu in nus:
        r = p / (1.0 + e * np.cos(nu))
        b = _gauss_b_aei(a, e, omega, float(nu), p, h, r)
        # max over 推力方向 = 各行 2 范数（单位推力下该根数变化率最大值）
        a_max = max(a_max, float(np.linalg.norm(b[0, :])))
        e_max = max(e_max, float(np.linalg.norm(b[1, :])))
        i_max = max(i_max, float(np.linalg.norm(b[2, :])))
    # 乘以 f（推力加速度幅值）
    return f * a_max, f * e_max, f * i_max


def _gauss_b_aei(
    a: float, e: float, omega: float, nu: float, p: float, h: float, r: float
) -> npt.NDArray[np.floating]:
    """Gauss 方程 3×3 子阵（a,e,i 行 × a_r,a_θ,a_h 列），RTN 系。

    圆轨道 e≈0 时 ė 行的 a_θ 项含 re→0、(p+r)cosν 仍有效；e 项系数含 e 但
    Q-law 在 e≈0 时不控 e（权重场景由调用方保证）。
    """
    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)
    cos_wv = np.cos(omega + nu)
    b = np.zeros((3, 3))
    # a 行
    b[0, 0] = (2.0 * a * a / h) * e * sin_nu
    b[0, 1] = (2.0 * a * a / h) * (p / r)
    # e 行
    b[1, 0] = (p / h) * sin_nu
    b[1, 1] = ((p + r) * cos_nu + r * e) / h
    # i 行（只有 a_h 列）
    b[2, 2] = r * cos_wv / h
    return b


def _q_value(
    oe: tuple[float, float, float],
    target: tuple[float, float, float],
    rates: tuple[float, float, float],
) -> float:
    """Q 函数值（只控 a,e,i）。oe/target = (a, e, i)，rates = (ȧ_max,ė_max,ị_max)。

    max_ν 由调用方预算（``_max_rates_aei``），避免梯度差分时重复扫描。
    """
    a, e, i_inc = oe
    a_t, e_t, i_t = target
    a_max, e_max, i_max = rates
    da = (a - a_t) / a_max if a_max > 0 else 0.0
    de = (e - e_t) / e_max if e_max > 0 else 0.0
    di = (i_inc - i_t) / i_max if i_max > 0 else 0.0
    return da * da + de * de + di * di


def _q_gradient(
    oe: tuple[float, float, float, float],
    target: tuple[float, float, float],
    f: float,
    mu: float,
) -> npt.NDArray[np.floating]:
    """∂Q/∂[a,e,i]（中心差分）。oe 含 ω 用于 max_ν。

    max_ν 在 base 点算一次，差分时视为常数（Q 对 X 的梯度主要来自 δ 项，
    max_ν 是缓变轨道几何量）。避免每个差分点重做 ν 扫描，性能提升 ~10x。
    """
    a, e, i_inc, omega = oe
    rates = _max_rates_aei(a, e, i_inc, omega, f, mu)
    eps_a = max(1e-3 * a, 1.0)
    eps_e = 1e-6
    eps_i = 1e-8
    eps = np.array([eps_a, eps_e, eps_i])
    grad = np.zeros(3)
    for k in range(3):
        op = (a + eps[0], e + eps[1] if k != 0 else e, i_inc + eps[2] if k == 2 else i_inc)
        # 简化：逐分量扰动
        if k == 0:
            op = (a + eps[0], e, i_inc)
            om = (a - eps[0], e, i_inc)
        elif k == 1:
            op = (a, e + eps[1], i_inc)
            om = (a, e - eps[1], i_inc)
        else:
            op = (a, e, i_inc + eps[2])
            om = (a, e, i_inc - eps[2])
        qp = _q_value(op, target, rates)
        qm = _q_value(om, target, rates)
        grad[k] = (qp - qm) / (2.0 * eps[k])
    return grad


def _direction_rtn_from_state(
    state7: npt.NDArray[np.floating],
    target_oe: tuple[float, float, float],
    mu: float,
    t_max: float,
) -> tuple[npt.NDArray[np.floating], float]:
    """单步 Q-law 推力方向（RTN 系单位向量）+ 加速度幅值 f=T/m。"""
    r_vec = state7[:3]
    v_vec = state7[3:6]
    m = float(state7[6])
    a, e, i_inc, _omega_raan, omega, nu = rv_to_keplerian(r_vec, v_vec, mu)

    f = t_max / m  # km/s²（t_max 是 N，m 是 kg，T/m 是 m/s² → 除以 1000 转 km）
    f_km = f / 1000.0

    grad = _q_gradient((a, e, i_inc, omega), target_oe, f_km, mu)
    p = a * (1.0 - e * e)
    h = float(np.sqrt(mu * p))
    r = float(np.linalg.norm(r_vec))
    b = _gauss_b_aei(a, e, omega, nu, p, h, r)

    mb = grad @ b  # (3,)
    norm_mb = float(np.linalg.norm(mb))
    if norm_mb < 1e-15:
        # 梯度近零（已接近目标），默认沿迹推力
        return np.array([0.0, 1.0, 0.0]), f_km
    # 最速下降方向：u_RTN = -BᵀMᵀ/||MB||
    u_rtn = -(b.T @ grad) / norm_mb
    u_norm = float(np.linalg.norm(u_rtn))
    if u_norm < 1e-15:
        return np.array([0.0, 1.0, 0.0]), f_km
    return u_rtn / u_norm, f_km


def _rtn_to_angles(u_rtn: npt.NDArray[np.floating]) -> tuple[float, float]:
    """RTN 单位向量 → 角度参数化 (θ₁, θ₂)（与 LowThrustShooting._angles_to_direction 对齐）。

    α(θ₁,θ₂) = [cosθ₁cosθ₂, sinθ₁cosθ₂, sinθ₂]。逆：θ₂=asin(α_z)，θ₁=atan2(α_y,α_x)。
    """
    theta2 = float(np.arcsin(np.clip(u_rtn[2], -1.0, 1.0)))
    theta1 = float(np.arctan2(u_rtn[1], u_rtn[0]))
    return theta1, theta2


def _rtn_to_inertial(
    u_rtn: npt.NDArray[np.floating],
    r_vec: npt.NDArray[np.floating],
    v_vec: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """RTN 方向向量 → 惯性系（propagate_compiled_lowthrust 接受惯性方向）。

    R=r/|r|，N=(r×v)/|r×v|，T=N×R。RTN 此处 R=径向、T=沿迹、N=法向；
    但 _gauss_b_aei 的列序是 (a_r, a_θ, a_h) = (径向, 横向/沿迹, 法向)，
    故惯性 = u_r*R + u_t*T + u_h*N。
    """
    r_hat = r_vec / np.linalg.norm(r_vec)
    h_vec = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec)
    if h_norm < 1e-15:
        return u_rtn[0] * r_hat  # 退化，仅径向
    n_hat = h_vec / h_norm
    t_hat = np.cross(n_hat, r_hat)  # 沿迹（横向）
    return u_rtn[0] * r_hat + u_rtn[1] * t_hat + u_rtn[2] * n_hat


def qlaw_guess(
    system: System,
    forces: Sequence[PhysicalModel],
    engine: EngineConfig,
    initial_state: npt.ArrayLike,
    initial_mass: float,
    target_oe: tuple[float, float, float],
    t0: float,
    tf: float,
    n_segments: int,
    *,
    step: float = 60.0,
    verbose: bool = False,
) -> tuple[np.ndarray, list[LowThrustSegment], np.ndarray, np.ndarray]:
    """Q-law 前向反馈积分，返回 (决策向量 y(3N), segments, Q 历史)。

    用 :func:`rk_step` 单步前向积分（每步重算 Q-law 方向，跟随轨道），eom 在
    Python 层（二体重力 + Q-law 推力）。最后把连续轨迹重采样成 ``n_segments``
    段常量控制（取每段中点时刻的方向），喂求解器。

    **为什么不用 propagate_compiled_lowthrust 接龙**：它段内固定惯性方向，长段
    里航天器转多圈，固定方向不跟随速度、平均不做功（实测 30 天固定切向 a 几乎
    不涨）。Q-law 需要每步跟随轨道重算方向，故用 rk_step 短步循环。

    Args:
        system: 动力学系统（取 origin 与中心体 μ）。
        forces: 非推力力模型（PointMassGravity 等，用于查 μ）。
        engine: 推进配置。
        initial_state: 出发状态 [r,v] (6,)。
        initial_mass: 初始质量 kg。
        target_oe: 目标 (a_T, e_T, i_T)（只控 a,e,i）。
        t0, tf: 起止时刻。
        n_segments: 重采样的段数（求解器决策变量数 = 3N）。
        step: 前向积分步长（秒），默认 60。方向跟随精度由它决定。
        verbose: 打印每段进度。

    Returns:
        (y, segments, q_history, final_state)：y 是求解器决策向量 (3N,)，segments
        是各段常量控制，q_history 是各段段中点的 Q 值，final_state 是 Q-law 前向
        积分的真实末态 7D（含质量），供测试验证根数收敛（注意：求解器用 y 重建
        会因段内固定方向而与 final_state 略有差异，final_state 是连续反馈的真实结果）。
    """
    from e2m2e.integrators import RkMethod, rk_step

    mu = _resolve_mu(system, forces)
    t_max = engine.t_max
    isp = engine.isp

    state = np.concatenate([np.asarray(initial_state, dtype=float), [float(initial_mass)]])
    g0 = _G0

    def eom(t: float, y: list[float]) -> list[float]:
        """二体重力 + Q-law 满推方向的 7D 右端项。"""
        y_arr = np.asarray(y, dtype=float)
        r_vec = y_arr[:3]
        v_vec = y_arr[3:6]
        m = float(y_arr[6])
        rn = float(np.linalg.norm(r_vec))
        a_grav = -mu / rn**3 * r_vec
        # Q-law 方向（RTN → 惯性），每步跟随轨道
        u_rtn, _ = _direction_rtn_from_state(y_arr, target_oe, mu, t_max)
        u_inertial = _rtn_to_inertial(u_rtn, r_vec, v_vec)
        a_thrust = (t_max / m) / 1000.0 * u_inertial  # km/s²
        mdot = -t_max / (isp * g0)
        return [
            v_vec[0],
            v_vec[1],
            v_vec[2],
            a_grav[0] + a_thrust[0],
            a_grav[1] + a_thrust[1],
            a_grav[2] + a_thrust[2],
            mdot,
        ]

    # 前向积分，记录轨迹（时间 + 状态）用于重采样
    times_rec: list[float] = [t0]
    states_rec: list[np.ndarray] = [state.copy()]
    t = t0
    h = step
    tol = 1e-10
    n_steps = 0
    while t < tf and n_steps < 2_000_000:
        n_steps += 1
        # 不越过 tf
        if t + h > tf:
            h = tf - t
        res = rk_step(RkMethod.PD45, t, state.tolist(), h, tol, eom, state_error_dim=6)  # type: ignore[arg-type]  # qlaw eom 签名 vs integrators wrapper 类型标注（预存）
        if res.error <= tol:
            t += h
            state = np.asarray(res.y_new, dtype=float)
            times_rec.append(t)
            states_rec.append(state.copy())
            h = float(res.h_next)
        else:
            h = float(res.h_next)
        if h < 1e-6:
            h = step  # 防步长坍缩

    # 重采样成 n_segments 段：每段取中点时刻的状态，算该时刻 Q-law 方向作段常量控制
    times_arr = np.asarray(times_rec)
    states_arr = np.asarray(states_rec)
    dt = (tf - t0) / n_segments
    y_parts: list[float] = []
    segments: list[LowThrustSegment] = []
    q_history: list[float] = []

    for seg in range(n_segments):
        t_mid = t0 + (seg + 0.5) * dt
        # 找最近的时间点（粗采样，够用）
        idx = int(np.argmin(np.abs(times_arr - t_mid)))
        mid_state = states_arr[idx]
        a, e, i_inc, omega_m, *_ = rv_to_keplerian(mid_state[:3], mid_state[3:6], mu)
        f_km = (t_max / mid_state[6]) / 1000.0
        rates_m = _max_rates_aei(a, e, i_inc, omega_m, f_km, mu)
        q_history.append(_q_value((a, e, i_inc), target_oe, rates_m))

        u_rtn, _ = _direction_rtn_from_state(mid_state, target_oe, mu, t_max)
        u_inertial = _rtn_to_inertial(u_rtn, mid_state[:3], mid_state[3:6])
        theta1, theta2 = _rtn_to_angles(u_inertial)
        y_parts.extend([1.0, theta1, theta2])
        segments.append(
            LowThrustSegment(throttle=1.0, direction=u_inertial / np.linalg.norm(u_inertial))
        )
        if verbose:
            print(
                f"seg {seg}: a={a:.1f} e={e:.4f} i={np.degrees(i_inc):.2f}° Q={q_history[-1]:.4e}"
            )

    return np.array(y_parts, dtype=float), segments, np.array(q_history), state


def _resolve_mu(system: object, forces: Sequence[PhysicalModel]) -> float:
    """从 PointMassGravity 或系统查中心体 μ。"""
    from ..forces import PointMassGravity

    for f in forces:
        if isinstance(f, PointMassGravity) and f.mu is not None:
            return float(f.mu)
    if hasattr(system, "gravitational_parameter"):
        origin = getattr(system, "origin", "EARTH")
        try:
            return float(system.gravitational_parameter(origin))  # type: ignore[arg-type]
        except Exception:
            pass
    # 纯二体测试兜底
    return Datum.DE440.earth_gm


def _estimate_h(dt: float) -> float:
    """段内初始步长：段长的 1/10，夹到 [1, dt]。"""
    return float(np.clip(dt / 10.0, 1.0, dt))


def make_shooter_for_qlaw(
    system: System,
    forces: Sequence[PhysicalModel],
    engine: EngineConfig,
    initial_state: npt.ArrayLike,
    initial_mass: float,
    target_state: npt.ArrayLike,
    t0: float,
    tf: float,
) -> LowThrustShooting:
    """便利构造：用同样参数建一个 LowThrustShooting，供 Q-law 初猜打磨。"""
    from .lowthrust_shooting import LowThrustShooting

    return LowThrustShooting(
        system, forces, engine, initial_state, initial_mass, target_state, t0, tf
    )


# SimpleNamespace 重导出供类型提示用（system 常是 SimpleNamespace）
__all__ = [
    "rv_to_keplerian",
    "qlaw_guess",
    "make_shooter_for_qlaw",
]
