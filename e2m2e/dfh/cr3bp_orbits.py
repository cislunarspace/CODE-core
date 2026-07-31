"""CR3BP 周期轨道生成：``design_orbit`` 链路的初猜段。

把 DFH 形状参数翻译成地月 CR3BP 周期轨道：

- DRO：以近侧 x 轴穿越点 ``x0`` 为族参数，从标准种子出发沿族行走，
  选取振幅（一个周期内距月球的最大距离，km）命中目标的成员；
- Halo：Richardson 三阶近似种子 + 定 ``z0`` 微分修正，沿族把 ``z0``
  走到目标面外振幅（km，符号区分北/南）；
- NRHO：Halo 族成员中按近月距（距月心 = 近月点高度 + 月球半径）选取，
  北/南对应 ``halo_class`` 0/1。

族行走统一用割线法（``_walk_family``）：前一条轨道的修正结果作为下
一条的初猜，天然延拓，避免大步长下微分修正发散。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..algorithms import DifferentialCorrection
from ..algorithms.halo_initial_guess import compute_halo_initial_guess
from ..core import CR3BP_Dynamics, CR3BP_System, Orbit

#: 地月 CR3BP 参数（与 examples/、tests/conftest.py 的标准系统一致）
EARTH_MOON_MU = 0.0121506683
CHAR_LENGTH_KM = 384400.0
CHAR_PERIOD_SEC = 27.32 * 86400.0

#: 月球平均半径（km），NRHO 近月点高度的起算面
MOON_RADIUS_KM = 1737.4

#: DRO 族标准种子（examples/orbit_design.py 验证过的初值）
_DRO_SEED_X0 = 0.79188556619742
_DRO_SEED_VY0 = 0.53682
_DRO_SEED_PERIOD = 3.0

#: Halo 族种子面外振幅（无量纲，小振幅下 Richardson 近似精度高）
_HALO_SEED_Z0 = 0.001

#: 固定 z0 行走的安全上界（按平动点）：Halo 族折叠点 L1 在 |z0|≈0.085、
#: L2 在 |z0|≈0.20（PAL 测试实测），固定 z0 的修正接近折叠点即失效，
#: 超过此值改用固定 x0 行走
_HALO_FOLD_Z0 = {1: 0.07, 2: 0.15}


class Cr3bpOrbitError(RuntimeError):
    """CR3BP 周期轨道生成失败（微分修正不收敛或族行走未命中目标）。"""


def earth_moon_system() -> CR3BP_System:
    """构造标准地月 CR3BP 系统（含特征尺度与平动点）。"""
    system = CR3BP_System(
        mu=EARTH_MOON_MU, primary="Earth", secondary="Moon"
    )._with_default_scales()
    system.set_characteristic_scales(CHAR_LENGTH_KM, CHAR_PERIOD_SEC)
    system.compute_libration_points()
    return system


def _moon_distance_minmax(
    dynamics: CR3BP_Dynamics, orbit: Orbit, n_points: int = 4000
) -> tuple[float, float]:
    """传播一个周期，返回距月心距离的最小/最大值（无量纲）。

    NRHO 近月点附近距离-时间曲线尖锐，240 点采样的极小值可偏离真值
    数十 km；4000 点把采样噪声压到亚 km 级（实测 6000 km 高度 NRHO
    约 0.3 km），族行走与测试断言用同一函数测量才一致。
    """
    mu = dynamics.system.mu
    t_eval = np.linspace(0.0, orbit.period, n_points)
    result = dynamics.propagate(orbit.states[0], (0.0, orbit.period), t_eval=t_eval)
    states = result["states"]
    dist = np.sqrt((states[:, 0] - (1.0 - mu)) ** 2 + states[:, 1] ** 2 + states[:, 2] ** 2)
    return float(dist.min()), float(dist.max())


def _correct_or_raise(corrector: DifferentialCorrection, guess: Orbit, label: str) -> Orbit:
    orbit = corrector.iterate_correction(initial_guess=guess, verbose=False)
    if orbit is None:
        raise Cr3bpOrbitError(f"{label} 微分修正未收敛: {corrector.termination_reason}")
    return orbit


def _walk_family(
    correct_at: Callable[[float, Orbit | None], Orbit],
    measure: Callable[[Orbit], float],
    target: float,
    p_seed: float,
    dp_init: float,
    *,
    max_step: float,
    tol: float,
    seed_orbit: Orbit | None = None,
    max_iter: int = 60,
) -> Orbit:
    """沿轨道族行走，使 ``measure(orbit(p))`` 命中 ``target``（±tol）。

    ``correct_at(p, guess)`` 在族参数 ``p`` 处修正出周期轨道；``measure``
    提取该轨道用于匹配目标的标量（振幅/近月距等）。假定 measure 随 p
    单调。先用一步试探出增减方向，定步行走跨过目标后用二分法收敛；
    修正发散时步长减半重试。``seed_orbit`` 给出时作为 ``p_seed`` 处
    已修正的轨道，跳过首次修正。
    """
    orbit = seed_orbit if seed_orbit is not None else correct_at(p_seed, None)
    p, m = p_seed, measure(orbit)
    if abs(m - target) <= tol:
        return orbit

    # 一步试探确定 measure 随 p 的增减方向
    probe = correct_at(p_seed + dp_init, orbit)
    m_probe = measure(probe)
    slope = (m_probe - m) / dp_init
    if slope == 0.0:
        raise Cr3bpOrbitError("族参数行走停滞：measure 不随参数变化")
    direction = 1.0 if (target - m) / slope > 0 else -1.0

    # 定步行走，跨过目标即得二分区间。
    # 修正失败（共振缝隙或越过族参数域边界，如 DRO 的 x0 撞月）时退回
    # 上一成功点并把步长减半——更靠近已知轨道，初猜更好；从失败点继续
    # 前进会在域边界处一路失败到步长耗尽。步长减到最小仍失败才判定
    # 不可达。
    step = max_step
    p_prev, m_prev, orbit_prev = p, m, orbit
    bracket: tuple[float, float, Orbit, Orbit] | None = None
    for _ in range(max_iter):
        p_try = p_prev + direction * step
        try:
            orbit_new = correct_at(p_try, orbit_prev)
        except Cr3bpOrbitError:
            step *= 0.5
            if step < 1e-4:
                raise Cr3bpOrbitError(f"族参数行走步长已减至最小仍未跨过目标 {target}") from None
            continue
        m_new = measure(orbit_new)
        if abs(m_new - target) <= tol:
            return orbit_new
        if (m_new - target) * (m_prev - target) <= 0:
            bracket = (p_prev, p_try, orbit_prev, orbit_new)
            break
        p_prev, m_prev, orbit_prev = p_try, m_new, orbit_new

    if bracket is None:
        raise Cr3bpOrbitError(f"族参数行走 {max_iter} 步内未跨过目标 {target}")

    # 二分收敛；中点失败时试 1/4、3/4 点（绕开共振缝隙）
    p_lo, p_hi, orbit_lo, orbit_hi = bracket
    m_lo = measure(orbit_lo)
    for _ in range(max_iter):
        p_mid = 0.5 * (p_lo + p_hi)
        orbit_mid = None
        m_mid = None
        for frac in (0.5, 0.25, 0.75):
            p_try = p_lo + frac * (p_hi - p_lo)
            guess = orbit_lo if abs(p_try - p_lo) <= abs(p_try - p_hi) else orbit_hi
            try:
                orbit_mid = correct_at(p_try, guess)
                p_mid = p_try
                m_mid = measure(orbit_mid)
                break
            except Cr3bpOrbitError:
                continue
        if orbit_mid is None or m_mid is None:
            raise Cr3bpOrbitError(
                f"二分区间内修正均失败（[{p_lo:.4f}, {p_hi:.4f}]），目标 {target} 不可达"
            )
        if abs(m_mid - target) <= tol:
            return orbit_mid
        if (m_mid - target) * (m_lo - target) > 0:
            p_lo, m_lo, orbit_lo = p_mid, m_mid, orbit_mid
        else:
            p_hi, orbit_hi = p_mid, orbit_mid

    raise Cr3bpOrbitError(f"族参数行走二分 {max_iter} 步内未命中目标 {target}")


def _correct_dro(dynamics: CR3BP_Dynamics, x0: float, guess: Orbit | None) -> Orbit:
    """在近侧 x 轴穿越点 ``x0`` 处修正 DRO（固定 x0，自由 vy0 与半周期）。"""
    if x0 >= 1.0 - dynamics.system.mu:
        # 穿越点越过月球（对侧），已不在近侧 DRO 族参数域内
        raise Cr3bpOrbitError(f"DRO x0={x0:.6f} 越过月心位置，超出族参数域")
    if guess is None:
        state = np.array([x0, 0.0, 0.0, 0.0, _DRO_SEED_VY0, 0.0])
        period = _DRO_SEED_PERIOD
    else:
        state = guess.states[0].copy()
        state[0] = x0
        period = guess.period
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0=x0)
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = period
    orbit = _correct_or_raise(corrector, seed, f"DRO(x0={x0:.6f})")
    if orbit.period > 1.2 * period:
        # 大步长行走时修正器会跳到长周期伪解（多圈对称周期轨道），周期
        # 相对初猜显著变长即判为伪解，交由族行走退半步重试
        raise Cr3bpOrbitError(
            f"DRO(x0={x0:.6f}) 修正跳到长周期伪解（T={orbit.period:.3f}，初猜 {period:.3f}）"
        )
    return orbit


def _correct_halo(
    dynamics: CR3BP_Dynamics, z0: float, libration_point: int, guess: Orbit | None
) -> Orbit:
    """在面外振幅 ``z0``（带符号，无量纲）处修正 Halo 轨道。"""
    if guess is None:
        halo_class = 0 if z0 > 0 else 1
        g = compute_halo_initial_guess(
            mu=dynamics.system.mu,
            z_amplitude=abs(z0),
            L=libration_point,
            halo_class=halo_class,
        )
        state = np.array([g["x0"], 0.0, z0, 0.0, g["vy0"], 0.0])
        period = 2.0 * g["T_half"]
    else:
        state = guess.states[0].copy()
        state[2] = z0
        period = guess.period
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_halo_orbit_fixed_z0(z0=z0, libration_point=libration_point)
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = period
    orbit = _correct_or_raise(corrector, seed, f"Halo(L{libration_point}, z0={z0:.6f})")
    if guess is not None and orbit.period > 1.2 * period:
        # 近月 NRHO 段 STM 条件数高，牛顿步易 overshoot 跳到长周期伪解，
        # 判为失败交由族行走退半步重试（同 ``_correct_dro`` 的处理）
        raise Cr3bpOrbitError(
            f"Halo(L{libration_point}, z0={z0:.6f}) 修正跳到长周期伪解"
            f"（T={orbit.period:.3f}，初猜 {period:.3f}）"
        )
    return orbit


def _correct_halo_x0(
    dynamics: CR3BP_Dynamics, x0: float, libration_point: int, guess: Orbit
) -> Orbit:
    """固定 x 穿越点 ``x0`` 修正 Halo 族轨道（折叠点附近的族参数）。

    固定 ``z0`` 的修正在 Halo 族折叠点（L2 约 |z0|≈0.17）前失效；
    折叠前后 ``x0`` 单调，改用它作族参数可一路走到 NRHO。
    """
    state = guess.states[0].copy()
    state[0] = x0
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_halo_orbit_fixed_x0(x0=x0, libration_point=libration_point)
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = guess.period
    return _correct_or_raise(corrector, seed, f"Halo(L{libration_point}, x0={x0:.6f})")


def _halo_seed_walk(
    dynamics: CR3BP_Dynamics,
    libration_point: int,
    z_sign: float,
) -> Orbit:
    """固定 ``z0`` 从 Richardson 种子走到 ``_HALO_FOLD_Z0``，作为固定
    ``x0`` 行走的出发成员。"""
    return _walk_family(
        correct_at=lambda z0, guess: _correct_halo(dynamics, z0, libration_point, guess),
        measure=lambda orbit: float(orbit.states[0, 2]),
        target=float(np.copysign(_HALO_FOLD_Z0[libration_point], z_sign)),
        p_seed=float(np.copysign(_HALO_SEED_Z0, z_sign)),
        dp_init=float(np.copysign(0.01, z_sign)),
        max_step=0.02,
        tol=1e-6,
    )


def design_dro(
    amplitude_km: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
    tol_km: float = 20.0,
) -> Orbit:
    """生成指定振幅的 DRO 周期轨道。

    振幅定义：一个周期内距月心距离最小/最大值的均值（km）。依据 DFH
    设计黄金样本标定：DFH 以 amplitude=10000 设计的 DRO 距月范围约
    4825~15290 km，(min+max)/2≈10058 km，比最大距离或 y 半宽更贴近
    输入值。以近侧 x 轴穿越点 ``x0`` 为族参数行走，命中 ``tol_km`` 内即停。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    du = dynamics.system.characteristic_length
    target_du = amplitude_km / du

    def measure(orbit: Orbit) -> float:
        d_min, d_max = _moon_distance_minmax(dynamics, orbit)
        return 0.5 * (d_min + d_max)

    return _walk_family(
        correct_at=lambda x0, guess: _correct_dro(dynamics, x0, guess),
        measure=measure,
        target=target_du,
        p_seed=_DRO_SEED_X0,
        dp_init=0.02,
        max_step=0.05,
        tol=tol_km / du,
    )


def design_halo(
    collinear_point: int,
    amplitude_km: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
) -> Orbit:
    """生成指定面外振幅的 Halo 周期轨道。

    ``amplitude_km`` 带符号：正为北族、负为南族（与 DFH ±73000 km 的
    约定一致）。振幅对应 Halo 参考状态（y=0 穿越点）的 z 坐标。
    |z0| 不超过 ``_HALO_FOLD_Z0`` 时直接固定 z0 行走；更大振幅先走到
    折叠点前的族成员，再改用固定 x0 行走逼近目标。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    du = dynamics.system.characteristic_length
    z_target = amplitude_km / du

    if abs(z_target) <= _HALO_FOLD_Z0[collinear_point]:
        return _walk_family(
            correct_at=lambda z0, guess: _correct_halo(dynamics, z0, collinear_point, guess),
            measure=lambda orbit: float(orbit.states[0, 2]),
            target=z_target,
            p_seed=float(np.copysign(_HALO_SEED_Z0, z_target)),
            dp_init=float(np.copysign(0.01, z_target)),
            max_step=0.02,
            tol=1e-6,
        )

    if collinear_point == 1:
        # L1 Halo 族在折叠点（|z0|≈0.085）后 z0 回落（转入 NRHO 分支），
        # 更大面外振幅的 L1 Halo 不存在；固定 x0 行走在 L1 会误入
        # 平面/垂直 Lyapunov 族，明确报错而不是返回错误的族
        raise Cr3bpOrbitError(
            f"L1 Halo 族面外振幅上限约 {_HALO_FOLD_Z0[1] * du:.0f} km（折叠点），"
            f"目标 {amplitude_km:.0f} km 不可达"
        )

    seed = _halo_seed_walk(dynamics, collinear_point, z_target)
    return _walk_family(
        correct_at=lambda x0, guess: _correct_halo_x0(dynamics, x0, collinear_point, guess),
        measure=lambda orbit: float(orbit.states[0, 2]),
        target=z_target,
        p_seed=float(seed.states[0, 0]),
        dp_init=-0.005,
        max_step=0.01,
        tol=1e-6,
        seed_orbit=seed,
    )


def _walk_pal_to_perilune(
    dynamics: CR3BP_Dynamics,
    libration_point: int,
    z_sign: float,
    target_du: float,
    tol_du: float,
    max_orbits: int = 600,
) -> Orbit:
    """L1 Halo 族的 NRHO 段：单次 PAL 延拓到目标近月距，再固定 z0 细化。

    L1 折叠点两侧固定 x0 修正均不收敛（会误入平面/垂直 Lyapunov 族），
    只能走 PAL；固定 z0 行走在近月 NRHO 段会因 STM 条件数高跳到长周期
    伪解（已由 ``_correct_halo`` 的伪解拒绝兜住，供二分细化退半步重试）。
    PAL 必须一次调用走全程：中途以新种子重启会重判延拓方向，在折叠点
    后方向来回翻转，实测分块重启在 z0≈∓0.13（近月距 ≈0.138 DU）处停滞，
    单次调用则可一路走到近月距 0.003 DU 以下。
    """
    from ..algorithms import Continuation

    seed = _correct_halo(dynamics, float(np.copysign(_HALO_SEED_Z0, z_sign)), libration_point, None)
    seed.family_type = "halo"
    seed.parameters = {
        "libration_point": libration_point,
        "halo_class": 0 if z_sign > 0 else 1,
        "amplitude_z": _HALO_SEED_Z0,
    }
    continuation = Continuation(corrector=DifferentialCorrection(dynamics))
    direction = "positive" if z_sign > 0 else "negative"

    family = continuation.halo_pseudo_arclength_continuation(
        seed_orbit=seed,
        n_orbits=max_orbits,
        direction=direction,
        step_size=0.0045,
        verbose=False,
    )
    prev_orbit = seed
    for orb in list(family.orbits)[1:]:
        perilune = _moon_distance_minmax(dynamics, orb)[0]
        if perilune <= target_du:
            # 跨过目标：以 z0 为参数在 NRHO 分支上二分细化
            return _walk_family(
                correct_at=lambda z0, guess: _correct_halo(dynamics, z0, libration_point, guess),
                measure=lambda o: _moon_distance_minmax(dynamics, o)[0],
                target=target_du,
                p_seed=float(orb.states[0, 2]),
                dp_init=float(orb.states[0, 2] - prev_orbit.states[0, 2]),
                max_step=0.005,
                tol=tol_du,
                seed_orbit=orb,
            )
        prev_orbit = orb

    raise Cr3bpOrbitError(f"PAL 延拓 {max_orbits} 条轨道仍未到达目标近月距 {target_du:.6f} DU")


def design_nrho(
    collinear_point: int,
    north_south: int,
    perilune_height_km: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
    tol_km: float = 10.0,
) -> Orbit:
    """生成指定近月点高度的 NRHO 周期轨道（Halo 族特选成员）。

    北/南（``north_south`` 1/2）对应 ``halo_class`` 0/1；近月距目标为
    ``perilune_height_km + MOON_RADIUS_KM``（距月心）。L2：先固定 z0
    走到折叠点前的族成员，再固定 x0 向月侧行走；L1：固定 x0 在折叠点
    两侧均失效，改用 PAL 延拓（``_walk_pal_to_perilune``）。近月距命中
    ``tol_km`` 内即停。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    du = dynamics.system.characteristic_length
    target_du = (perilune_height_km + MOON_RADIUS_KM) / du
    z_sign = 1.0 if north_south == 1 else -1.0

    if collinear_point == 1:
        orbit = _walk_pal_to_perilune(dynamics, 1, z_sign, target_du, tol_km / du)
        if orbit.states[0, 2] * z_sign < 0.0:
            # L1 NRHO 分支上微分修正的参考状态可能落在远月侧 xz 穿越点
            # （z 与北/南约定反号）；同一周期轨道平移半周期即得近月侧
            # 穿越点，z 符号与北/南约定一致
            state_half = np.asarray(
                dynamics.propagate_orbit_state_at_time(orbit, orbit.period / 2.0),
                dtype=float,
            )
            rephased = Orbit(
                states=state_half.reshape(1, -1),
                times=np.array([0.0]),
                system=dynamics.system,
            )
            rephased.period = orbit.period
            orbit = rephased
        return orbit

    seed = _halo_seed_walk(dynamics, collinear_point, z_sign)

    def measure(orbit: Orbit) -> float:
        return _moon_distance_minmax(dynamics, orbit)[0]

    return _walk_family(
        correct_at=lambda x0, guess: _correct_halo_x0(dynamics, x0, collinear_point, guess),
        measure=measure,
        target=target_du,
        p_seed=float(seed.states[0, 0]),
        dp_init=-0.005,
        max_step=0.01,
        tol=tol_km / du,
        seed_orbit=seed,
    )
