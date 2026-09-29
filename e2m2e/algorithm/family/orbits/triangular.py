"""三角平动点族：L4/L5 邻域的 SPO/LPO/Horseshoe 周期轨道。

SPO（短周期族）周期约 1 朔望月、近稳定；LPO（长周期族）不稳定，
大振幅呈马蹄形（Horseshoe 是 LPO 的大振幅成员分类，不获第二套
求解器，ADR 0028）。三者共用 ``_design_triangular_family`` 后端
（Rust 种子 + 全周期 PAL 行走）。
"""

from __future__ import annotations

import numpy as np

from ....data.types.orbit import Orbit
from ....integrators import orbit_family_metric_py
from ...dynamics import CR3BP_Dynamics
from ...results import FamilyGenerationResult
from ...solver.differential_correction import DifferentialCorrection
from ..lpo_initial_guess import compute_lpo_initial_guess
from ..spo_initial_guess import compute_spo_initial_guess
from ..strategies import lpo_fixed_x0, spo_fixed_x0
from ..triangular_initial_guess import compute_triangular_initial_guess
from .walk import (
    Cr3bpOrbitError,
    _generate_rust_family,
    _validate_amplitude_range,
    earth_moon_system,
)


def _correct_spo(
    dynamics: CR3BP_Dynamics,
    x0: float,
    libration_point: int,
    guess: Orbit | None,
) -> Orbit:
    """在 x₀ 处修正 SPO（通用平面周期修正，无对称性假设）。

    SPO 是 L4/L5 短周期族成员，xy 平面内周期轨道，不具有 x 轴
    对称性（y₀≠0）。使用 iterate_full_period_correction 做全周期闭合。

    首次调用（guess=None）使用短周期模态线性化初猜（不含长/垂直模态）。
    后续调用保留已收敛轨道状态作初猜。
    """
    if guess is None:
        # 从线性化短周期模态构造初猜（小振幅初猜对 Newton 收敛足够近）
        state, period = compute_spo_initial_guess(
            dynamics.system,
            libration_point,
            amplitude_km=1000.0,
        )
        # 覆盖 x₀ 到族参数指定值
        state[0] = x0
    else:
        state = guess.states[0].copy()
        state[0] = x0
        state[2] = 0.0
        state[5] = 0.0
        assert guess.period is not None
        period = guess.period

    corrector = DifferentialCorrection(dynamics)
    corrector.configure(spo_fixed_x0(x0=x0, libration_point=libration_point))
    seed = Orbit(
        states=state.reshape(1, -1),
        times=np.array([0.0]),
        system=dynamics.system,
    )
    seed.period = period

    result = corrector.iterate_full_period_correction(seed, verbose=False)
    orbit = result.orbit
    if orbit is None:
        raise Cr3bpOrbitError(
            f"SPO(L{libration_point}, x0={x0:.6f}) 全周期修正未收敛: {result.message}"
        )
    assert orbit.period is not None
    if orbit.period < 1.0 or orbit.period > 15.0:
        raise Cr3bpOrbitError(
            f"SPO(L{libration_point}, x0={x0:.6f}) 周期异常（T={orbit.period:.3f}，预期 1.0-15.0）"
        )
    return orbit


def _l45_distance(
    dynamics: CR3BP_Dynamics,
    orbit: Orbit,
    point: int,
    n_points: int = 2000,
) -> tuple[float, float]:
    """传播一个周期，返回距 L4/L5 径向距离的最小/最大值（无量纲）。"""
    assert orbit.period is not None
    minimum, maximum = orbit_family_metric_py(
        float(dynamics.system.mu),
        "l45-distance",
        point,
        orbit.states[0],
        float(orbit.period),
        sample_count=n_points,
    )
    return float(minimum), float(maximum)


def design_spo(
    libration_point: int,
    amplitude_km: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
    tol_km: float = 20.0,
) -> Orbit:
    r"""生成指定振幅的 L4/L5 SPO 周期轨道。

    SPO（Short-Period Orbit）是 CR3BP 中围绕三角平动点的短周期族
    成员（Gómez vol II, $\mathcal{L}_s$），周期约 1 朔望月（~28 天），
    近稳定（特征值模 ≈ 1.001）。

    振幅定义：一个周期内距 L4/L5 径向距离最小/最大值的均值（km）。
    以 x₀ 为族参数，在 L4/L5 附近做二分搜索逼近目标振幅。

    实现策略（直接二分，非族行走）：
    SPO 短周期族在 L4/L5 附近的振幅-×₀ 映射非单调（小振幅区间
    步长敏感），_walk_family 假设单调性，不适合此族。改用 x₀
    上的直接二分搜索 + 每步全周期修正，每步都从线性化初猜出发
    （SPO 近稳定，牛顿收敛可靠）。

    Args:
        libration_point: 平动点编号（4=L4, 5=L5）。
        amplitude_km: 目标振幅（km）。
        dynamics: CR3BP 动力学对象；缺省构造标准地月系统。
        tol_km: 振幅匹配容差（km），默认 20。

    Returns:
        修正后的 SPO 周期轨道。

    References:
        Gómez et al. (2001). Dynamics and mission design near libration
        points, Vol. II. ESA Contract Report.
        Capdevila & Howell (2018). A transfer network linking Earth,
        Moon, and the triangular libration point regions. JGCD.
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    du = dynamics.system.characteristic_length
    assert du is not None
    mu = dynamics.system.mu
    target_du = amplitude_km / du
    tol_du = tol_km / du

    def measure(orbit: Orbit) -> float:
        d_min, d_max = _l45_distance(dynamics, orbit, libration_point)
        return 0.5 * (d_min + d_max)

    lp_x = 0.5 - mu

    # 预计算一条种子轨道（L4/L5 附近），后续二分步复用作初猜
    seed_orbit = _correct_spo(dynamics, lp_x, libration_point, None)

    def _correct_with_seed(x0: float) -> Orbit:
        """用种子轨道作初猜的修正（比从线性化初猜快很多）。"""
        return _correct_spo(dynamics, x0, libration_point, seed_orbit)

    # 二分搜索 x₀：x₀ 越远离 L4（向月侧减小），振幅越大
    # 搜索范围：x₀ ∈ [lp_x - 0.05, lp_x + 0.02]
    x_lo = lp_x - 0.05  # 大振幅端
    x_hi = lp_x + 0.02  # 小振幅端

    best_orbit = None
    best_err = float("inf")

    for _ in range(25):
        x_mid = 0.5 * (x_lo + x_hi)
        try:
            orbit_mid = _correct_with_seed(x_mid)
        except Cr3bpOrbitError:
            x_hi = x_mid
            continue
        amp_mid = measure(orbit_mid)
        err = abs(amp_mid - target_du)
        if err < best_err:
            best_err = err
            best_orbit = orbit_mid
        if err <= tol_du:
            return orbit_mid
        # 振幅随 x₀ 递减（x₀ 增大 → 更靠近 L4 → 振幅减小）
        if amp_mid > target_du:
            x_lo = x_mid
        else:
            x_hi = x_mid

    if best_orbit is not None and best_err <= 5 * tol_du:
        return best_orbit

    # 回退：小振幅（< 种子振幅）直接从线性化初猜修正
    # 种子轨道在 lp_x 处振幅约 1500 km；目标更小时直接用线性化
    # 初猜（不含长/垂直模态）在合适 x₀ 处修正
    seed_amp = measure(seed_orbit)
    if target_du < seed_amp:
        # 在 lp_x 和 lp_x+0.01 之间找合适 x₀
        for dx in np.linspace(0.0, 0.010, 11):
            try:
                orbit_try = _correct_spo(dynamics, lp_x + dx, libration_point, None)
                if abs(measure(orbit_try) - target_du) <= 5 * tol_du:
                    return orbit_try
            except Cr3bpOrbitError:
                continue

    raise Cr3bpOrbitError(
        f"SPO(L{libration_point}, amp={amplitude_km:.0f} km) 未命中目标"
        f"（最佳误差 {best_err * du:.0f} km）"
    )


def _correct_lpo(
    dynamics: CR3BP_Dynamics,
    x0: float,
    libration_point: int,
    guess: Orbit | None,
) -> Orbit:
    """在 x₀ 处修正 LPO（通用平面周期修正，无对称性假设）。

    与 _correct_spo 同构，但使用长周期模态初猜。
    LPO 不稳定（λ≈1.8），修正器需要更严格的收敛条件。

    大振幅 LPO 呈马蹄形（Horseshoe），跨越 L4-L1-L5
    （Marchal 1990, Brown 猜想 C.2）。

    首次调用（guess=None）使用线性化长周期模态初猜。
    后续调用保留已收敛轨道状态作初猜。
    """
    if guess is None:
        # 从线性化长周期模态构造初猜（小振幅初猜对 Newton 收敛足够近）
        state, period = compute_lpo_initial_guess(
            dynamics.system,
            libration_point,
            amplitude_km=1000.0,
        )
        # 覆盖 x₀ 到族参数指定值
        state[0] = x0
    else:
        state = guess.states[0].copy()
        state[0] = x0
        state[2] = 0.0
        state[5] = 0.0
        assert guess.period is not None
        period = guess.period

    # 搜索阶段放宽积分步长上限：默认 0.01 TU 对长周期 LPO
    # （~21 TU/圈）意味着每次 STM 传播 ≥2100 步，网格搜索 45 点 × 修正
    # 迭代 ~700 次传播共 ~110s。rtol=1e-12 自适应仍控制积分精度，max_step
    # 只是上限：0.1 TU 下实测 2.5x 加速、收敛产物一致（振幅 49994 vs
    # 50000 km、周期同 21.124 TU）。结束恢复，不污染调用方共享 dynamics。
    _orig_max_step = dynamics.max_step
    dynamics.max_step = max(dynamics.max_step, 0.1)
    try:
        corrector = DifferentialCorrection(dynamics)
        corrector.configure(lpo_fixed_x0(x0=x0, libration_point=libration_point))
        seed = Orbit(
            states=state.reshape(1, -1),
            times=np.array([0.0]),
            system=dynamics.system,
        )
        seed.period = period

        result = corrector.iterate_full_period_correction(seed, verbose=False)
        orbit = result.orbit
        if orbit is None:
            raise Cr3bpOrbitError(
                f"LPO(L{libration_point}, x0={x0:.6f}) 全周期修正未收敛: {result.message}"
            )
        assert orbit.period is not None
        # LPO 周期范围：极限 21.07 nd（~91 天），大振幅可到 ~30+ nd
        if orbit.period < 10.0 or orbit.period > 50.0:
            raise Cr3bpOrbitError(
                f"LPO(L{libration_point}, x0={x0:.6f}) 周期异常"
                f"（T={orbit.period:.3f}，预期 10.0-50.0）"
            )
        return orbit
    finally:
        dynamics.max_step = _orig_max_step


def design_lpo(
    libration_point: int,
    amplitude_km: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
    tol_km: float = 20.0,
) -> Orbit:
    r"""生成指定振幅的 L4/L5 LPO 周期轨道。

    LPO（Long-Period Orbit）是 CR3BP 中围绕三角平动点的长周期族
    成员（Gómez vol II, $\mathcal{L}_l$）。小振幅时为椭圆形，大振幅
    时呈马蹄形（Horseshoe），跨越 L4-L1-L5（Marchal 1990, Brown C.2）。

    振幅定义同 SPO：一个周期内距 L4/L5 径向距离最小/最大值的均值（km）。
    以 x₀ 为族参数，用网格搜索 + 局部精化逼近目标振幅。

    实现策略（网格搜索 + 局部二分）：
    LPO 长周期族的振幅-x₀ 映射高度非单调（小振幅椭圆 → 混沌过渡区 →
    大振幅马蹄族），简单二分搜索无法收敛。改用两步策略：
    1) 均匀网格采样 x₀，找到振幅最接近目标的候选点；
    2) 在候选点附近做局部二分精化（利用局部单调性）。

    Args:
        libration_point: 平动点编号（4=L4, 5=L5）。
        amplitude_km: 目标振幅（km），范围 1,000~110,000。
        dynamics: CR3BP 动力学对象；缺省构造标准地月系统。
        tol_km: 振幅匹配容差（km），默认 20。

    Returns:
        修正后的 LPO 周期轨道（小振幅椭圆 或 大振幅马蹄形）。

    References:
        Gómez et al. (2001). Vol. II. 长周期族 L_l。
        Marchal (1990). The Three-Body Problem. Brown 猜想 C.2。
        Taylor (1981). A&A 103, 288. 马蹄周期轨道数值计算。
    """
    if not 1000.0 <= amplitude_km <= 110000.0:
        raise ValueError(f"LPO amplitude 应在 1000~110000 km 之间，实际为 {amplitude_km:.0f} km")
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    du = dynamics.system.characteristic_length
    assert du is not None
    mu = dynamics.system.mu
    target_du = amplitude_km / du
    tol_du = tol_km / du

    def measure(orbit: Orbit) -> float:
        d_min, d_max = _l45_distance(dynamics, orbit, libration_point)
        return 0.5 * (d_min + d_max)

    lp_x = 0.5 - mu

    # LPO 长周期族的振幅-x₀ 映射高度非单调（小振幅椭圆族 → 混沌过渡区 →
    # 大振幅马蹄族），不能用简单二分搜索。改用分层网格搜索 + 局部精化：
    # 1) 粗网格（20 点）找到振幅最接近目标的候选区间
    # 2) 细网格（10 点）在候选区间内精化
    # 3) 局部二分搜索（10 步）做最终精化
    x_lo = lp_x - 0.20  # 大振幅端（马蹄方向）
    x_hi = lp_x + 0.05  # 小振幅端

    def _grid_search(
        x_lo: float, x_hi: float, n_pts: int, seed: Orbit | None
    ) -> tuple[float | None, Orbit | None, float]:
        """在 [x_lo, x_hi] 均匀采样 n_pts 点，返回最佳 (x0, orbit, err)。"""
        b_x0: float | None = None
        b_orb: Orbit | None = None
        b_err = float("inf")
        for x0 in np.linspace(x_lo, x_hi, n_pts):
            try:
                orb = _correct_lpo(dynamics, x0, libration_point, seed)
            except Cr3bpOrbitError:
                continue
            amp = measure(orb)
            err = abs(amp - target_du)
            if err < b_err:
                b_err = err
                b_x0 = x0
                b_orb = orb
            if err <= tol_du:
                return x0, orb, err
        return b_x0, b_orb, b_err

    # 第 1 步：粗网格搜索（30 点）
    best_x0, best_orbit, best_err = _grid_search(x_lo, x_hi, 30, None)

    if best_orbit is None:
        raise Cr3bpOrbitError(
            f"LPO(L{libration_point}, amp={amplitude_km:.0f} km) 网格搜索无收敛轨道"
        )
    if best_err <= tol_du:
        return best_orbit

    # 第 2 步：细网格精化（在最佳候选点 ±2 步长内，15 点）
    dx = (x_hi - x_lo) / 30
    assert best_x0 is not None  # best_orbit 非空时 best_x0 必非空
    refine_lo = max(x_lo, best_x0 - 2 * dx)
    refine_hi = min(x_hi, best_x0 + 2 * dx)
    rx0, rorb, rerr = _grid_search(refine_lo, refine_hi, 15, best_orbit)
    if rorb is not None and rerr < best_err:
        best_x0, best_orbit, best_err = rx0, rorb, rerr
    if best_err <= tol_du:
        return best_orbit

    # 第 3 步：局部二分精化（在最佳候选点 ±1 步长内）
    if best_x0 is not None:
        refine_lo = max(x_lo, best_x0 - dx)
        refine_hi = min(x_hi, best_x0 + dx)
        seed_orbit = best_orbit

        for _ in range(10):
            x_mid = 0.5 * (refine_lo + refine_hi)
            try:
                orbit_mid = _correct_lpo(dynamics, x_mid, libration_point, seed_orbit)
            except Cr3bpOrbitError:
                break
            amp_mid = measure(orbit_mid)
            err = abs(amp_mid - target_du)
            if err < best_err:
                best_err = err
                best_orbit = orbit_mid
                seed_orbit = orbit_mid
            if err <= tol_du:
                return orbit_mid
            if amp_mid > target_du:
                refine_lo = x_mid
            else:
                refine_hi = x_mid

    # LPO 振幅-x₀ 映射高度非单调（混沌过渡区），网格搜索可能无法
    # 精确命中容差。放宽回退阈值：max(10*tol, 1000 km) 允许大振幅
    # 区域的合理误差。
    fallback_tol = max(10 * tol_du, 1000.0 / du)
    if best_err <= fallback_tol:
        return best_orbit

    raise Cr3bpOrbitError(
        f"LPO(L{libration_point}, amp={amplitude_km:.0f} km) 未命中目标"
        f"（最佳误差 {best_err * du:.0f} km）"
    )


def design_horseshoe(
    libration_point: int,
    amplitude_km: float = 100000.0,
    *,
    dynamics: CR3BP_Dynamics | None = None,
    tol_km: float = 50.0,
) -> Orbit:
    r"""生成 L4/L5 Horseshoe 马蹄形周期轨道。

    Horseshoe 是 LPO 长周期族的大振幅成员，轨道形状呈马蹄形，
    跨越 L4-L1-L5（Marchal 1990, Brown 猜想 C.2, Taylor 1981）。

    本函数是 design_lpo 的便捷封装，默认振幅为 100,000 km。
    振幅定义同 LPO/SPO：距 L4/L5 径向距离均值（km）。

    Args:
        libration_point: 平动点编号（4=L4, 5=L5）。
        amplitude_km: 目标振幅（km），范围 50,000~110,000，默认 100,000。
        dynamics: CR3BP 动力学对象；缺省构造标准地月系统。
        tol_km: 振幅匹配容差（km），默认 50（比 LPO 默认 20 宽松，
            因为大振幅族行走精度下降）。

    Returns:
        修正后的 Horseshoe 周期轨道。

    References:
        Taylor (1981). A&A 103, 288. Sun-Jupiter 马蹄周期轨道。
        Marchal (1990). The Three-Body Problem. Brown C.2 证实。
        Murray & Dermott (1999). §3.9 Horseshoe 运动学描述。
    """
    if not 50000.0 <= amplitude_km <= 110000.0:
        raise ValueError(
            f"Horseshoe amplitude 应在 50000~110000 km 之间，实际为 {amplitude_km:.0f} km"
        )
    return design_lpo(
        libration_point,
        amplitude_km,
        dynamics=dynamics,
        tol_km=tol_km,
    )


def design_triangular(
    point: int,
    amplitude_in_km: float,
    amplitude_out_km: float,
    phase_in: float,
    phase_out: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
) -> Orbit:
    """生成 L4/L5 邻域拟周期轨道初猜（等边三角形平动点）。

    面内振幅默认均分给短/长两模态（拆分比例待 golden 标定）。不做微分
    修正：现有 ``algorithms/strategies/`` 全基于 x 轴/y 轴镜面对称，L4/L5
    无适用对称性。初猜状态直接作星历修正 patch points 的采样基准。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    state0, nominal_period = compute_triangular_initial_guess(
        dynamics.system,
        point,
        amplitude_in_km,
        amplitude_out_km,
        phase_in,
        phase_out,
    )
    orbit = Orbit(
        states=state0.reshape(1, -1),
        times=np.array([0.0]),
        system=dynamics.system,
    )
    orbit.period = nominal_period
    orbit.family_type = f"L{point}"
    orbit.parameters = {
        "point": point,
        "amplitude_in_km": amplitude_in_km,
        "amplitude_out_km": amplitude_out_km,
    }
    return orbit


def _design_triangular_family(
    family_type: str,
    libration_point: int,
    min_amplitude_km: float,
    max_amplitude_km: float,
    *,
    n_orbits: int,
    continuation_direction: str,
    match_tolerance_km: float,
    dynamics: CR3BP_Dynamics,
) -> FamilyGenerationResult:
    """三角平动点周期族（SPO/LPO）生成：小振幅种子 + 全周期 PAL 行走。

    Rust 在 L4/L5 邻域构造并修正种子（SPO 短周期模态 / LPO 长周期模态），
    按请求的 increase/decrease-x0 初始方向做平面全周期 PAL（ADR 0028）；
    收集振幅（距 L4/L5
    径向距离 min/max
    均值，与 ``design_spo``/``design_lpo`` 同一定义）落入
    ``[min_amplitude_km, max_amplitude_km]`` 的成员，至多 ``n_orbits``
    条。PAL 只调用一次，方向只决定首步切向量；完整扫描该弧长链后再按
    振幅范围和匹配容差筛选，因此不要求 x0、周期或振幅全局单调，也能
    保留转向后重入范围的成员。PAL 链断裂或估算步数未覆盖请求上界时，
    通过 :class:`FamilyGenerationResult` 返回部分族及状态。
    """
    if family_type not in ("spo", "lpo", "horseshoe"):
        raise ValueError(f"family_type 必须为 spo、lpo 或 horseshoe，当前为 {family_type!r}")
    if libration_point not in (4, 5):
        raise ValueError(f"libration_point 必须为 4 或 5，当前为 {libration_point}")
    if n_orbits < 1:
        raise ValueError(f"n_orbits 必须大于 0，当前为 {n_orbits}")
    _validate_amplitude_range(min_amplitude_km, max_amplitude_km)
    if continuation_direction not in ("decrease-x0", "increase-x0"):
        raise ValueError("三角族 continuation_direction 必须为 'decrease-x0' 或 'increase-x0'")
    if match_tolerance_km <= 0.0:
        raise ValueError("match_tolerance_km 必须为正数")
    return _generate_rust_family(
        family_type,
        libration_point,
        n_orbits,
        dynamics,
        min_amplitude_km=min_amplitude_km,
        max_amplitude_km=max_amplitude_km,
        continuation_direction=continuation_direction,
        match_tolerance_km=match_tolerance_km,
    )


def design_spo_family(
    libration_point: int,
    min_amplitude_km: float,
    max_amplitude_km: float,
    *,
    n_orbits: int = 50,
    continuation_direction: str = "decrease-x0",
    match_tolerance_km: float = 20.0,
    dynamics: CR3BP_Dynamics | None = None,
) -> FamilyGenerationResult:
    """生成 L4/L5 SPO 族：短周期族中振幅落入请求范围的成员。

    振幅定义同 ``design_spo``（距 L4/L5 径向距离 min/max 均值，km）。
    族生成方法见 ``_design_triangular_family``。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    return _design_triangular_family(
        "spo",
        libration_point,
        min_amplitude_km,
        max_amplitude_km,
        n_orbits=n_orbits,
        continuation_direction=continuation_direction,
        match_tolerance_km=match_tolerance_km,
        dynamics=dynamics,
    )


def design_lpo_family(
    libration_point: int,
    min_amplitude_km: float,
    max_amplitude_km: float,
    *,
    n_orbits: int = 50,
    continuation_direction: str = "decrease-x0",
    match_tolerance_km: float = 20.0,
    dynamics: CR3BP_Dynamics | None = None,
) -> FamilyGenerationResult:
    """生成 L4/L5 LPO 族：长周期族中振幅落入请求范围的成员。

    振幅定义同 ``design_lpo``。族生成方法见 ``_design_triangular_family``。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    return _design_triangular_family(
        "lpo",
        libration_point,
        min_amplitude_km,
        max_amplitude_km,
        n_orbits=n_orbits,
        continuation_direction=continuation_direction,
        match_tolerance_km=match_tolerance_km,
        dynamics=dynamics,
    )


def design_horseshoe_family(
    libration_point: int,
    min_amplitude_km: float = 50000.0,
    max_amplitude_km: float = 110000.0,
    *,
    n_orbits: int = 50,
    continuation_direction: str = "decrease-x0",
    match_tolerance_km: float = 50.0,
    dynamics: CR3BP_Dynamics | None = None,
) -> FamilyGenerationResult:
    """生成 L4/L5 Horseshoe 族：LPO 长周期族的大振幅（马蹄形）成员。

    Horseshoe 是 LPO 族的成员分类，不获第二套求解器（ADR 0028）：
    沿 LPO 链行走到大振幅段，收集振幅落入请求范围的成员并标记为
    ``horseshoe``。声明范围不得超出已标定的可达包络。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    return _design_triangular_family(
        "horseshoe",
        libration_point,
        min_amplitude_km,
        max_amplitude_km,
        n_orbits=n_orbits,
        continuation_direction=continuation_direction,
        match_tolerance_km=match_tolerance_km,
        dynamics=dynamics,
    )
