"""Axial 轨道初猜（Gómez Type B 分岔族）。

Axial 轨道从 planar Lyapunov 轨道通过 pitchfork 分岔产生（Gómez Type B），
关于 x 轴对称。与 Halo（Type A, xz 平面对称）的区别：

- Halo: 初始 (x₀, 0, z₀, 0, ẏ₀, 0) — 从 xz 平面出发，ż₀=0
- Axial: 初始 (x₀, 0, 0, 0, ẏ₀, ż₀) — 从 xy 平面出发，z₀=0，ż₀≠0

分岔机制：沿 planar Lyapunov 族行走，在垂直临界轨道处（monodromy 的
z-vz 块半迹 vt = +1），Lyapunov 轨道的面内振幅不再为零。Axial 族从此
分岔，继承面内振幅，并叠加面外速度 ż₀。

初猜构造：
1. 沿 Lyapunov 族扫描，找到 vt = +1 且 Jacobi 在 Haapala 区间的轨道
2. 取该 Lyapunov 轨道的初始状态 (x₀, 0, 0, 0, ẏ₀, 0)
3. 叠加小 ż₀ 扰动：(x₀, 0, 0, 0, ẏ₀, ż₀_seed)

References:
    Gómez, G., et al. (2001). Dynamics and mission design near libration
    points, Vol. III, Sec. 3.3. Type B 垂直临界分岔。
    Haapala & Howell (2016). Axial Jacobi 范围 [2.991, 3.021]（L1）、
    [2.967, 3.014]（L2）。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ...data.templates.enums import LibrationPoint
from ...data.types.orbit import Orbit
from ..dynamics import CR3BP_Dynamics
from ..solver.differential_correction import DifferentialCorrection
from .lissajous_initial_guess import _linear_modes

#: 共线点编号 → LibrationPoint 枚举
_COLLINEAR = {1: LibrationPoint.L1, 2: LibrationPoint.L2, 3: LibrationPoint.L3}

#: Haapala & Howell (2016) Axial Jacobi 区间（用于筛选正确的分岔点）
_AXIAL_C_RANGES: dict[int, tuple[float, float]] = {
    1: (2.991, 3.021),
    2: (2.967, 3.014),
}

#: 模块级缓存：(mu, L) → (lyapunov_state0, lyapunov_period)
_bifurcation_cache: dict[tuple[float, int], tuple[np.ndarray, float]] = {}


def _correct_lyapunov_fixed_x0(
    dynamics: CR3BP_Dynamics,
    x0: float,
    guess: Orbit | None,
    *,
    x_L: float,
    x_factor: float,
    vy_factor: float,
    T_lin: float,
) -> Orbit | None:
    """固定 x₀ 修正一条 planar Lyapunov 轨道（x 轴对称）。

    首次调用（guess=None）时用线性化特征向量构造种子；后续调用用上一条
    轨道做自然延拓。
    """
    if guess is None:
        delta = x0 - x_L
        alpha = delta / x_factor
        vy0 = alpha * vy_factor
        state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        period = T_lin
    else:
        state = guess.states[0].copy()
        state[0] = x0
        state[2] = 0.0
        state[5] = 0.0
        assert guess.period is not None
        period = guess.period

    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0=x0)
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = period
    return corrector.iterate_correction(initial_guess=seed, verbose=False).orbit


def _correct_lyapunov_fixed_t(
    dynamics: CR3BP_Dynamics,
    t_half: float,
    guess: Orbit,
) -> Orbit | None:
    """固定半周期修正一条 planar Lyapunov 轨道。

    固定 T_half 时自由变量为 x₀ 和 ẏ₀，绕开了固定 x₀ 步进时在同一 x₀
    处存在多个周期解（Lyapunov vs DRO）导致跳支的问题。
    """
    state = guess.states[0].copy()
    state[2] = 0.0
    state[5] = 0.0
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_t(t_half=t_half)
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = 2.0 * t_half
    return corrector.iterate_correction(initial_guess=seed, verbose=False).orbit


def _vertical_trace(dynamics: CR3BP_Dynamics, orbit: Orbit) -> float:
    """计算 monodromy 的 z-vz 块半迹 vt = 0.5*(M[2,2] + M[5,5])。"""
    assert orbit.period is not None
    M = dynamics.compute_state_transition_matrix(orbit.states[0], orbit.period)
    return 0.5 * (M[2, 2] + M[5, 5])


def _find_axial_bifurcation_seed(
    dynamics: CR3BP_Dynamics,
    libration_point: int,
) -> tuple[npt.NDArray[np.floating], float]:
    """沿 planar Lyapunov 族扫描垂直临界轨道，返回 Axial 分岔种子。

    扫描策略：

    1. **种子**：在平动点附近 (x₀ = x_L - 0.01) 用线性化特征向量构造
       种子，修正出第一条 Lyapunov 轨道。
    2. **T 延拓**：固定半周期 T_half（自由变量 x₀ 和 ẏ₀），逐步增加 T，
       用上一条轨道做自然延拓。这绕开了固定 x₀ 步进时在同一 x₀ 处存在
       多个周期解（Lyapunov vs DRO）导致跳支的问题。

    每条轨道计算 monodromy 的 z-vz 块半迹 vt。找到 vt 跨过 +1 且
    Jacobi 落在 Haapala Axial 区间的分岔点，用二分法精化。

    Returns:
        (state0, period)：分岔点处 Lyapunov 轨道的初始状态
        (x₀, 0, 0, 0, ẏ₀, 0) 与周期。
    """
    key = (dynamics.system.mu, libration_point)
    if key in _bifurcation_cache:
        return _bifurcation_cache[key]

    system = dynamics.system
    if not system.has_L_points:
        system.compute_libration_points()

    # 线性化参数
    omega_xy, v_xy, _, _, x_L = _linear_modes(system, libration_point)
    T_lin = 2.0 * np.pi / omega_xy

    # 线性化种子：从面内特征向量导出 x₀-ẏ₀ 关系
    re_vy = float(np.real(v_xy[1]))
    im_vy = float(np.imag(v_xy[1]))
    phi = np.arctan2(re_vy, im_vy)  # 使 y(0)=0 的相位
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    x_factor = float(np.real(v_xy[0]) * cos_phi - np.imag(v_xy[0]) * sin_phi)
    vy_factor = float(np.real(v_xy[4]) * cos_phi - np.imag(v_xy[4]) * sin_phi)

    # Jacobi 筛选区间
    C_lo, C_hi = _AXIAL_C_RANGES.get(libration_point, (2.5, 3.5))

    # ---- 种子轨道：x₀ = x_L - 0.01 修正一条 Lyapunov 轨道 ----
    x0_seed = x_L - 0.01
    seed_orbit = _correct_lyapunov_fixed_x0(
        dynamics,
        x0_seed,
        None,
        x_L=x_L,
        x_factor=x_factor,
        vy_factor=vy_factor,
        T_lin=T_lin,
    )
    if seed_orbit is None:
        raise RuntimeError(f"L{libration_point} Lyapunov 种子轨道修正失败 (x₀={x0_seed:.4f})")

    # ---- T 延拓扫描 ----
    scan: list[tuple[float, Orbit, float, float]] = []  # (T, orbit, vt, C)

    # 记录种子轨道
    assert seed_orbit.period is not None
    vt0 = _vertical_trace(dynamics, seed_orbit)
    C0 = system.get_jacobi_constant(seed_orbit.states[0])
    scan.append((float(seed_orbit.period), seed_orbit, vt0, C0))

    prev_orbit = seed_orbit
    t_step = 0.02
    t_step_min = 0.002

    for _ in range(500):
        assert prev_orbit.period is not None
        T_new = prev_orbit.period + t_step
        orbit = _correct_lyapunov_fixed_t(dynamics, T_new / 2.0, prev_orbit)

        if orbit is None:
            t_step *= 0.5
            if t_step < t_step_min:
                break
            continue

        vt = _vertical_trace(dynamics, orbit)
        C = system.get_jacobi_constant(orbit.states[0])

        # 跳支检测：vt 突变（远大于平滑延拓的每步变化）说明修正器收敛到了
        # 另一条轨道支——L2 平面 Lyapunov 族大振幅处与近月轨道族交互，
        # 实测 0.02 步长一步 vt 0.97→0.14、C 3.17→2.92（跳到近月支）；
        # 平滑延拓下 0.02 步 vt 变化 ~5e-3，阈值 0.3 远高于正常波动，
        # 对 L1（平滑）无影响。跳变时丢弃该轨道、减小步长从 prev 重试。
        prev_vt = scan[-1][2]
        if abs(vt - prev_vt) > 0.3:
            t_step *= 0.5
            if t_step < t_step_min:
                break
            continue

        assert orbit.period is not None
        scan.append((float(orbit.period), orbit, vt, C))

        prev_orbit = orbit
        t_step = 0.02  # 恢复步长

        # C 已远低于 Axial 区间，无需继续
        if C_lo - 0.05 > C:
            break

    # ---- 找 vt=+1 穿越并按 Jacobi 筛选 ----
    crossings: list[tuple[int, int]] = []  # scan index pairs (lo, hi)
    for j in range(1, len(scan)):
        vt0 = scan[j - 1][2]
        vt1 = scan[j][2]
        if (vt0 - 1.0) * (vt1 - 1.0) <= 0:
            crossings.append((j - 1, j))

    if not crossings:
        raise RuntimeError(f"L{libration_point} Lyapunov 族未找到 vt=+1 穿越点")

    # 选 Jacobi 最接近 Axial 区间中心的穿越
    C_center = 0.5 * (C_lo + C_hi)
    best_pair = min(
        crossings,
        key=lambda p: min(
            abs(scan[p[0]][3] - C_center),
            abs(scan[p[1]][3] - C_center),
        ),
    )

    # ---- 二分精化 ----
    T_lo, orbit_lo, vt_lo, _ = scan[best_pair[0]]
    T_hi, orbit_hi, _, _ = scan[best_pair[1]]

    for _ in range(30):
        T_mid = 0.5 * (T_lo + T_hi)
        guess = orbit_lo if abs(T_mid - T_lo) < abs(T_mid - T_hi) else orbit_hi
        orbit_mid = _correct_lyapunov_fixed_t(dynamics, T_mid / 2.0, guess)
        if orbit_mid is None:
            break
        vt_mid = _vertical_trace(dynamics, orbit_mid)
        if abs(vt_mid - 1.0) < 1e-4:
            orbit_lo = orbit_mid
            break
        if (vt_mid - 1.0) * (vt_lo - 1.0) > 0:
            T_lo, orbit_lo, vt_lo = T_mid, orbit_mid, vt_mid
        else:
            T_hi, orbit_hi = T_mid, orbit_mid

    assert orbit_lo.period is not None
    result = (orbit_lo.states[0].copy(), float(orbit_lo.period))
    _bifurcation_cache[key] = result
    return result


def compute_axial_initial_guess(
    dynamics: CR3BP_Dynamics,
    collinear_point: int,
    vz0: float,
) -> tuple[npt.NDArray[np.floating], float]:
    """构造 Axial 轨道初猜状态。

    从 planar Lyapunov 族的垂直临界轨道（Gómez Type B 分岔点）出发，
    继承 Lyapunov 轨道的面内振幅 (x₀, ẏ₀)，叠加小 ż₀ 扰动。

    Args:
        dynamics: CR3BP 动力学对象（用于 Lyapunov 族扫描）。
        collinear_point: 共线点编号 1/2/3。
        vz0: 初始 z 方向速度（无量纲 DO/TU），带符号区分上/下族。

    Returns:
        (state0, period)：t=0 的 6 维状态 (x₀, 0, 0, 0, ẏ₀, ż₀)
        与分岔 Lyapunov 轨道的周期（无量纲 TU）。
    """
    lyapunov_state, lyapunov_period = _find_axial_bifurcation_seed(dynamics, collinear_point)

    # Axial 种子 = Lyapunov 状态 + ż₀ 扰动
    state0 = lyapunov_state.copy()
    state0[2] = 0.0  # z₀ = 0（xy 平面出发）
    state0[5] = vz0  # ż₀ 扰动

    return state0, lyapunov_period
