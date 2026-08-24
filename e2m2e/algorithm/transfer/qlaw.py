"""Q-law 低推力初猜生成器（Lyapunov 反馈律）。

用 Q-law（Petropoulos；Holt 2024 式 6-10）做低推力转移的初猜生成：前向反馈
积分产出次优控制历史，喂 :class:`~e2m2e.algorithm.transfer.lowthrust_shooting.LowThrustShooting`
做解析雅可比打磨。是 gap-analysis 中 Q-law 作初猜、打靶优化两级流程的上半段。

## 最简版（控 a, e, i）

只控半长轴、偏心率、倾角，规避 ω̇/Ω̇ 的 1/e、1/sin i 奇异。Q 函数：

```
Q = Σ_{X∈{a,e,i}} (δ(X, X_T) / max_ν(Ẋ))²
```

控制律（Holt 式 8，最速下降）：``u_RTN = −f·BᵀMᵀ/||MB||``，其中 f=T/m、
B 是 Gauss 方程 3×3 子阵、M=∂Q/∂[a,e,i]（中心差分）。油门固定满推。

## 架构：Rust 反馈积分 + Python 初猜组装

Rust 内核在一次调用中完成逐步重算方向的 Q-law 自适应反馈积分与 Q 函数评估。
Python 侧只解析参数、从 Rust 轨迹选择段中点并组装求解器需要的控制段，不保留
Python 数值降级路径；独立公开的 :func:`rv_to_keplerian` 维持既有兼容行为，
不参与反馈积分。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..forces import PhysicalModel
from .lowthrust_shooting import EngineConfig, LowThrustSegment

if TYPE_CHECKING:
    from ..dynamics import System
    from .lowthrust_shooting import LowThrustShooting


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


def _rtn_to_angles(u_rtn: npt.NDArray[np.floating]) -> tuple[float, float]:
    """RTN 单位向量 → 角度参数化 (θ₁, θ₂)（与 LowThrustShooting._angles_to_direction 对齐）。

    α(θ₁,θ₂) = [cosθ₁cosθ₂, sinθ₁cosθ₂, sinθ₂]。逆：θ₂=asin(α_z)，θ₁=atan2(α_y,α_x)。
    """
    theta2 = float(np.arcsin(np.clip(u_rtn[2], -1.0, 1.0)))
    theta1 = float(np.arctan2(u_rtn[1], u_rtn[0]))
    return theta1, theta2


def _assemble_guess(
    times_arr: npt.NDArray[np.floating],
    states_arr: npt.NDArray[np.floating],
    target_oe: tuple[float, float, float],
    mu: float,
    t_max: float,
    t0: float,
    tf: float,
    n_segments: int,
    verbose: bool,
) -> tuple[np.ndarray, list[LowThrustSegment], np.ndarray]:
    """把连续反馈轨迹重采样为求解器使用的分段常量初猜。"""
    dt = (tf - t0) / n_segments
    y_parts: list[float] = []
    segments: list[LowThrustSegment] = []
    q_history: list[float] = []

    for seg in range(n_segments):
        t_mid = t0 + (seg + 0.5) * dt
        idx = int(np.argmin(np.abs(times_arr - t_mid)))
        mid_state = states_arr[idx]
        from e2m2e.integrators import qlaw_segment_direction_py, require_rust_extension

        require_rust_extension("qlaw_segment_direction_py")
        result = qlaw_segment_direction_py(mid_state.tolist(), list(target_oe), mu, t_max)
        a = float(result["a"])
        e = float(result["e"])
        i_inc = float(result["i"])
        q_history.append(float(result["q_value"]))
        u_inertial = np.asarray(result["u_inertial"], dtype=float)

        theta1, theta2 = _rtn_to_angles(u_inertial)
        y_parts.extend([1.0, theta1, theta2])
        segments.append(
            LowThrustSegment(throttle=1.0, direction=u_inertial / np.linalg.norm(u_inertial))
        )
        if verbose:
            print(
                f"seg {seg}: a={a:.1f} e={e:.4f} i={np.degrees(i_inc):.2f}° Q={q_history[-1]:.4e}"
            )

    return np.array(y_parts, dtype=float), segments, np.array(q_history)


def _qlaw_guess_rust(
    mu: float,
    engine: EngineConfig,
    initial_state: npt.ArrayLike,
    initial_mass: float,
    target_oe: tuple[float, float, float],
    t0: float,
    tf: float,
    n_segments: int,
    step: float,
    verbose: bool,
) -> tuple[np.ndarray, list[LowThrustSegment], np.ndarray, np.ndarray]:
    """调用 Rust 反馈积分，再由 Python 组装分段初猜。"""
    from e2m2e.integrators import qlaw_propagate_py, require_rust_extension

    require_rust_extension("qlaw_propagate_py", "qlaw_segment_direction_py")
    initial_state7 = np.concatenate([np.asarray(initial_state, dtype=float), [float(initial_mass)]])
    result = qlaw_propagate_py(
        t0,
        tf,
        initial_state7.tolist(),
        list(target_oe),
        mu,
        engine.t_max,
        engine.isp,
        step,
        1e-10,
        2_000_000,
    )
    times_arr = np.asarray(result["time"], dtype=float)
    states_arr = np.asarray(result["states"], dtype=float)
    y, segments, q_history = _assemble_guess(
        times_arr,
        states_arr,
        target_oe,
        mu,
        engine.t_max,
        t0,
        tf,
        n_segments,
        verbose,
    )
    return y, segments, q_history, states_arr[-1].copy()


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

    Q-law 反馈积分、方向和 Q 函数评估由 Rust 内核一次完成；Python 侧只把
    Rust 轨迹重采样为 ``n_segments`` 段常量控制，喂给求解器。

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
    from e2m2e.integrators import require_rust_extension

    require_rust_extension("qlaw_propagate_py", "qlaw_segment_direction_py")
    mu = _resolve_mu(system, forces)
    return _qlaw_guess_rust(
        mu,
        engine,
        initial_state,
        initial_mass,
        target_oe,
        t0,
        tf,
        n_segments,
        step,
        verbose,
    )


def _resolve_mu(system: object, forces: Sequence[PhysicalModel]) -> float:
    """从 PointMassGravity 或系统查中心体 μ；解析失败时抛异常。

    μ 是动力学核心参数，查不到即报错，不默认回退（#352）。
    """
    from ..forces import PointMassGravity

    for f in forces:
        if isinstance(f, PointMassGravity) and f.mu is not None:
            return float(f.mu)
    if hasattr(system, "gravitational_parameter"):
        origin = getattr(system, "origin", "EARTH")
        return float(system.gravitational_parameter(origin))  # type: ignore[arg-type]
    raise RuntimeError(
        "Q-law: 无法解析中心体 μ（forces 无 PointMassGravity.mu，"
        "system 无 gravitational_parameter）"
    )


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
