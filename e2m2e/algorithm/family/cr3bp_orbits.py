"""CR3BP 周期轨道生成：``design_orbit`` 链路的初猜段。

把 DFH 形状参数翻译成地月 CR3BP 周期轨道：

- DRO：以近侧 x 轴穿越点 ``x0`` 为族参数，从标准种子出发沿族行走，
  选取振幅（一个周期内距月球的最大距离，km）命中目标的成员；
- DPO：与 DRO 对称的顺行族，以 ``x0`` 为族参数，vy0 < 0（顺行），
  振幅定义同 DRO（距月心距离 min/max 均值，km）；
- Halo：Richardson 三阶近似种子 + 定 ``z0`` 微分修正，沿族把 ``z0``
  走到目标面外振幅（km，符号区分北/南）；
- NRHO：Halo 族成员中按近月距（距月心 = 近月点高度 + 月球半径）选取，
  北/南对应 ``halo_class`` 0/1；
- Axial：Gómez Type B 分岔族，以面外速度 ``vz0`` 为族参数，
  从 xy 平面出发的 3D 周期轨道（x 轴对称），振幅 = max|z|（km）。

族行走统一用割线法（``_walk_family``）：前一条轨道的修正结果作为下
一条的初猜，天然延拓，避免大步长下微分修正发散。
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from ...data.templates import ConvergenceState, FailureCause
from ...data.templates.seed import (  # noqa: F401
    _AXIAL_SEED_VZ0,
    _DPO_SEED_PERIOD,
    _DPO_SEED_VY0,
    _DPO_SEED_X0,
    _DRO_SEED_PERIOD,
    _DRO_SEED_VY0,
    _DRO_SEED_X0,
    _HALO_FOLD_Z0,
    _HALO_SEED_Z0,
    _SPO_L4_SEED_VX0,
    _SPO_L4_SEED_VY0,
    _SPO_L4_SEED_X0,
    _SPO_L4_SEED_Y0,
    _SPO_SEED_PERIOD,
    CHAR_LENGTH_KM,
    CHAR_PERIOD_SEC,
    EARTH_MOON_MU,
    MOON_RADIUS_KM,
)
from ...data.types.orbit import Orbit, OrbitFamily
from ...integrators import orbit_family_metric_py
from ..dynamics import CR3BP_Dynamics, CR3BP_System
from ..results import FamilyGenerationResult, ResultStatus
from ..solver.differential_correction import DifferentialCorrection
from .axial_initial_guess import compute_axial_initial_guess
from .halo_family import halo_pseudo_arclength_continuation
from .halo_initial_guess import compute_halo_initial_guess
from .lissajous_initial_guess import (
    compute_lissajous_bounded_trajectory,
    compute_lissajous_initial_guess,
)
from .lpo_initial_guess import compute_lpo_initial_guess
from .spo_initial_guess import compute_spo_initial_guess
from .triangular_initial_guess import compute_triangular_initial_guess


class Cr3bpOrbitError(RuntimeError):
    """CR3BP 轨道生成硬失败，携带统一状态三元组。"""

    def __init__(
        self,
        message: str,
        *,
        status: ConvergenceState = ConvergenceState.FAILED,
        cause: FailureCause = FailureCause.UNKNOWN,
    ) -> None:
        super().__init__(message)
        ResultStatus(status, cause, message)
        self.status = status
        self.cause = cause
        self.message = message


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
    assert orbit.period is not None  # 周期轨道必有 period
    minimum, maximum = orbit_family_metric_py(
        float(dynamics.system.mu),
        "moon-distance",
        0,
        orbit.states[0],
        float(orbit.period),
        sample_count=n_points,
    )
    return float(minimum), float(maximum)


def _correct_or_raise(corrector: DifferentialCorrection, guess: Orbit, label: str) -> Orbit:
    result = corrector.iterate_correction(initial_guess=guess, verbose=False)
    if result.orbit is None:
        raise Cr3bpOrbitError(f"{label} 微分修正未收敛: {result.message}")
    return result.orbit


def _require_orbit(o: Orbit | None) -> Orbit:
    """运行时守卫：seed_orbit 保证 guess 不为 None。"""
    assert o is not None
    return o


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
    """沿轨道族行走，使 ``measure(orbit(p))`` 命中 ``target`` （±tol）。

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
        assert guess.period is not None
        period = guess.period
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0=x0)
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = period
    orbit = _correct_or_raise(corrector, seed, f"DRO(x0={x0:.6f})")
    assert orbit.period is not None
    if orbit.period > 1.2 * period:
        # 大步长行走时修正器会跳到长周期伪解（多圈对称周期轨道），周期
        # 相对初猜显著变长即判为伪解，交由族行走退半步重试
        raise Cr3bpOrbitError(
            f"DRO(x0={x0:.6f}) 修正跳到长周期伪解（T={orbit.period:.3f}，初猜 {period:.3f}）"
        )
    return orbit


def _correct_dpo(dynamics: CR3BP_Dynamics, x0: float, guess: Orbit | None) -> Orbit:
    """在近侧 x 轴穿越点 ``x0`` 处修正 DPO（固定 x0，自由 vy0 与半周期）。

    DPO 与 DRO 使用相同修正策略（``setup_2D_symmetric_x_fixed_x0``），
    但 vy0 < 0（顺行）。种子从反转 DRO vy0 的微分修正收敛获得。

    DPO 族不稳定，族行走时 vy0 与 x0 的映射关系比 DRO 更非线性。
    首次调用（guess=None）使用种子常量；后续调用保留已收敛轨道的完整
    状态作为初猜（不仅覆盖 x0），减少不稳定族上的跳支。

    固定 x0 的对称修正对同一 x0 同时存在顺行与逆行两支解，初猜落在
    逆行支会收敛到逆行解；保留完整状态作初猜并不能杜绝（#587，打包
    baseline dpo 前 4 成员曾因此为逆行）。收敛后校验月心角动量顺行
    （h_z > 0），不满足按伪解同路径拒绝，交由族行走退半步重试。
    """
    if guess is None:
        state = np.array([x0, 0.0, 0.0, 0.0, _DPO_SEED_VY0, 0.0])
        period = _DPO_SEED_PERIOD
    else:
        state = guess.states[0].copy()
        state[0] = x0
        assert guess.period is not None
        period = guess.period
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0=x0)
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = period
    orbit = _correct_or_raise(corrector, seed, f"DPO(x0={x0:.6f})")
    assert orbit.period is not None
    # DPO 族不稳定，周期变化幅度比 DRO 大；放宽伪解阈值以避免误杀
    # 正常族行走中周期跳变到 2 倍以上才是真伪解（多圈对称周期轨道）
    if orbit.period > 2.0 * period:
        raise Cr3bpOrbitError(
            f"DPO(x0={x0:.6f}) 修正跳到长周期伪解（T={orbit.period:.3f}，初猜 {period:.3f}）"
        )
    # 顺行分支校验（#587）：穿越点月心角动量 h_z = (x − x_moon)·vy 必须
    # 顺行（近侧 x0 < x_moon 即 vy0 < 0）；逆行支按伪解同路径拒绝
    x_moon = 1.0 - dynamics.system.mu
    h_z = (float(orbit.states[0, 0]) - x_moon) * float(orbit.states[0, 4])
    if h_z <= 0.0:
        raise Cr3bpOrbitError(
            f"DPO(x0={x0:.6f}) 修正收敛到逆行支"
            f"（vy0={float(orbit.states[0, 4]):+.4f}，h_z={h_z:+.5f}）"
        )
    return orbit


def _correct_halo(
    dynamics: CR3BP_Dynamics, z0: float, libration_point: int, guess: Orbit | None
) -> Orbit:
    """在面外振幅 ``z0`` （带符号，无量纲）处修正 Halo 轨道。"""
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
        assert guess.period is not None
        period = guess.period
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_halo_orbit_fixed_z0(z0=z0, libration_point=libration_point)
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = period
    orbit = _correct_or_raise(corrector, seed, f"Halo(L{libration_point}, z0={z0:.6f})")
    assert orbit.period is not None
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

    固定 ``z0`` 的修正在 Halo 族折叠点（L2 约 ``|z0|≈0.17``）前失效；
    折叠前后 ``x0`` 单调，改用它作族参数可一路走到 NRHO。
    """
    state = guess.states[0].copy()
    state[0] = x0
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_halo_orbit_fixed_x0(x0=x0, libration_point=libration_point)
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = guess.period
    return _correct_or_raise(corrector, seed, f"Halo(L{libration_point}, x0={x0:.6f})")


def _correct_axial(
    dynamics: CR3BP_Dynamics, vz0: float, libration_point: int, guess: Orbit | None
) -> Orbit:
    """在面外速度 ``vz0`` （带符号，无量纲）处修正 Axial 轨道（Type B）。

    Axial 轨道的初始状态为 (x0, 0, 0, 0, y_dot0, vz0)，利用 x 轴
    对称性做半周期修正（约束 y=0, z=0, x_dot=0 at T/2）。
    """
    if guess is None:
        state0, period = compute_axial_initial_guess(
            dynamics,
            collinear_point=libration_point,
            vz0=vz0,
        )
        state = state0
    else:
        state = guess.states[0].copy()
        state[2] = 0.0  # z0 = 0（x 轴上）
        state[5] = vz0
        assert guess.period is not None
        period = guess.period
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_axial_orbit_fixed_vz0(vz0=vz0, libration_point=libration_point)
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = period
    orbit = _correct_or_raise(corrector, seed, f"Axial(L{libration_point}, vz0={vz0:.6f})")
    assert orbit.period is not None
    if guess is not None and orbit.period > 1.2 * period:
        raise Cr3bpOrbitError(
            f"Axial(L{libration_point}, vz0={vz0:.6f}) 修正跳到长周期伪解"
            f"（T={orbit.period:.3f}，初猜 {period:.3f}）"
        )
    return orbit


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

    振幅定义：一个周期内距月心距离最小/最大值的均值（km），比最大距离
    或 y 半宽更贴近输入值（如 amplitude=10000 的 DRO 距月范围约
    4825~15290 km，(min+max)/2≈10058 km）。以近侧 x 轴穿越点 ``x0``
    为族参数行走，命中 ``tol_km`` 内即停。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    du = dynamics.system.characteristic_length
    assert du is not None
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


def design_dpo(
    amplitude_km: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
    tol_km: float = 20.0,
) -> Orbit:
    """生成指定振幅的 DPO（Direct Prograde Orbit）周期轨道。

    DPO 是 xy 平面内围绕月球的顺行周期轨道（旋转坐标系下逆时针），
    与 DRO（逆行）对称。振幅定义同 DRO：一个周期内距月心距离
    最小/最大值的均值（km）。以近侧 x 轴穿越点 ``x0`` 为族参数行走，
    命中 ``tol_km`` 内即停。

    References:
        Folta et al. (2015). An Earth–Moon system trajectory design
        reference catalog. AIAA SciTech.
        Guzzetti et al. (2016). Rapid trajectory design in the
        Earth–Moon ephemeris system via an interactive catalog of
        periodic orbits. JGCD.
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    du = dynamics.system.characteristic_length
    assert du is not None
    target_du = amplitude_km / du

    def measure(orbit: Orbit) -> float:
        d_min, d_max = _moon_distance_minmax(dynamics, orbit)
        return 0.5 * (d_min + d_max)

    return _walk_family(
        correct_at=lambda x0, guess: _correct_dpo(dynamics, x0, guess),
        measure=measure,
        target=target_du,
        p_seed=_DPO_SEED_X0,
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
    ``|z0|`` 不超过 ``_HALO_FOLD_Z0`` 时直接固定 z0 行走；更大振幅先走到
    折叠点前的族成员，再改用固定 x0 行走逼近目标。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    du = dynamics.system.characteristic_length
    assert du is not None
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
        correct_at=lambda x0, guess: _correct_halo_x0(
            dynamics, x0, collinear_point, _require_orbit(guess)
        ),
        measure=lambda orbit: float(orbit.states[0, 2]),
        target=z_target,
        p_seed=float(seed.states[0, 0]),
        dp_init=-0.005,
        max_step=0.01,
        tol=1e-6,
        seed_orbit=seed,
    )


def design_halo_family(
    libration_point: int,
    max_amplitude_km: float,
    *,
    n_orbits: int = 50,
    dynamics: CR3BP_Dynamics | None = None,
) -> OrbitFamily | FamilyGenerationResult:
    """生成一族 Halo 周期轨道（从种子延拓到指定面外振幅上限）。

    族从 Richardson 小振幅种子（``_HALO_SEED_Z0``）出发，自然参数
    延拓（固定 ``z0`` 微分修正）覆盖 ``[0, |max_amplitude_km|]``，
    至多 ``n_orbits`` 条（含种子）。振幅上限不得超过该平动点的族
    折叠点（``seed._HALO_FOLD_Z0``），折叠点后固定 z0 延拓失效。

    ``max_amplitude_km`` 带符号：正为北族、负为南族（与 ``design_halo``
    约定一致）；符号只决定延拓方向，振幅上限取绝对值。

    Args:
        libration_point: 平动点编号（1=L1, 2=L2）。
        max_amplitude_km: 族振幅上限（km），带符号区分北/南族。
        n_orbits: 族成员数量上限（含种子），默认 50。
        dynamics: CR3BP 动力学；缺省构造标准地月系统。

    Returns:
        含一族周期轨道的 :class:`OrbitFamily`。
    """
    if libration_point not in (1, 2):
        raise ValueError(f"libration_point 必须为 1 或 2，当前为 {libration_point}")
    if n_orbits < 1:
        raise ValueError(f"n_orbits 必须大于 0，当前为 {n_orbits}")
    if max_amplitude_km == 0:
        raise ValueError("max_amplitude_km 不能为 0")
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    from .rust_generation import generate_rust_family

    result = generate_rust_family(
        "halo",
        libration_point,
        n_orbits,
        dynamics,
        max_amplitude_km=max_amplitude_km,
    )
    if result.status is ConvergenceState.CONVERGED:
        return result.family
    return result


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
    from ..solver.continuation import Continuation

    seed = _correct_halo(dynamics, float(np.copysign(_HALO_SEED_Z0, z_sign)), libration_point, None)
    seed.family_type = "halo"
    seed.parameters = {
        "libration_point": libration_point,
        "halo_class": 0 if z_sign > 0 else 1,
        "amplitude_z": _HALO_SEED_Z0,
    }
    continuation = Continuation(corrector=DifferentialCorrection(dynamics))
    direction = "positive" if z_sign > 0 else "negative"

    family = halo_pseudo_arclength_continuation(
        continuation,
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
    ``perilune_height_km + MOON_RADIUS_KM`` （距月心）。L2：先固定 z0
    走到折叠点前的族成员，再固定 x0 向月侧行走；L1：固定 x0 在折叠点
    两侧均失效，改用 PAL 延拓（``_walk_pal_to_perilune``）。近月距命中
    ``tol_km`` 内即停。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    du = dynamics.system.characteristic_length
    assert du is not None
    target_du = (perilune_height_km + MOON_RADIUS_KM) / du
    z_sign = 1.0 if north_south == 1 else -1.0

    if collinear_point == 1:
        orbit = _walk_pal_to_perilune(dynamics, 1, z_sign, target_du, tol_km / du)
        if orbit.states[0, 2] * z_sign < 0.0:
            # L1 NRHO 分支上微分修正的参考状态可能落在远月侧 xz 穿越点
            # （z 与北/南约定反号）；同一周期轨道平移半周期即得近月侧
            # 穿越点，z 符号与北/南约定一致
            assert orbit.period is not None
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
        correct_at=lambda x0, guess: _correct_halo_x0(
            dynamics, x0, collinear_point, _require_orbit(guess)
        ),
        measure=measure,
        target=target_du,
        p_seed=float(seed.states[0, 0]),
        dp_init=-0.005,
        max_step=0.01,
        tol=tol_km / du,
        seed_orbit=seed,
    )


def design_lissajous(
    collinear_point: int,
    amplitude_in_km: float,
    amplitude_out_km: float,
    phase_in: float,
    phase_out: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
    n_periods: int = 3,
) -> Orbit:
    """生成指定共线点（L1/L2/L3）的 Lissajous 拟周期轨道。

    Lissajous 面内/面外频率不可约，是准周期轨道，不做周期闭合。一阶线性
    初猜在非线性 CR3BP 下必发散（不稳定方向泄漏，见
    :func:`compute_lissajous_bounded_trajectory`），故本函数返回**中心流形
    约化流的有界多点轨迹**（覆盖 ``n_periods`` 个名义周期），供下游可视化
    与 patch-point 采样直接使用——原生 CR3BP 重传播会重新激发不稳定方向
    而发散，下游不得用 ``states[0]`` 重传播。

    ``period`` 取面内名义周期 2π/ω_xy；``states[0]`` 为历元参考状态。

    Args:
        collinear_point: 共线点编号 1/2/3。
        amplitude_in_km / amplitude_out_km: 面内/面外振幅（km）。
        phase_in / phase_out: 面内/面外初始相位（0~1）。
        dynamics: CR3BP 动力学；缺省构造标准地月系统。
        n_periods: 返回轨迹覆盖的名义周期数（默认 3）。

    Returns:
        含多点有界轨迹的 :class:`Orbit`。中心流形约化失败时退回一阶线性
        单点初猜并发出 :class:`RuntimeWarning` （保下游不崩，但失去有界性）。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())

    try:
        states, times, nominal_period = compute_lissajous_bounded_trajectory(
            dynamics.system,
            collinear_point,
            amplitude_in_km,
            amplitude_out_km,
            phase_in,
            phase_out,
            n_periods=n_periods,
        )
    except RuntimeError as exc:
        # 非典型参数下中心流形约化可能失败：退回线性单点初猜，明确告警。
        # 下游此时不应依赖有界性（patch-point 采样仍会发散）。
        warnings.warn(
            f"中心流形约化失败，退回一阶线性 Lissajous 初猜（无有界性保证）：{exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        state0, nominal_period = compute_lissajous_initial_guess(
            dynamics.system,
            collinear_point,
            amplitude_in_km,
            amplitude_out_km,
            phase_in,
            phase_out,
        )
        states = state0.reshape(1, -1)
        times = np.array([0.0])

    orbit = Orbit(states=states, times=times, system=dynamics.system)
    orbit.period = nominal_period
    orbit.family_type = "lissajous"
    orbit.parameters = {
        "collinear_point": collinear_point,
        "amplitude_in_km": amplitude_in_km,
        "amplitude_out_km": amplitude_out_km,
    }
    return orbit


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


def _z_amplitude_max(dynamics: CR3BP_Dynamics, orbit: Orbit, n_points: int = 1000) -> float:
    """传播一个周期，返回 |z| 的最大值（无量纲 DU）。"""
    assert orbit.period is not None
    _, maximum = orbit_family_metric_py(
        float(dynamics.system.mu),
        "z-amplitude",
        0,
        orbit.states[0],
        float(orbit.period),
        sample_count=n_points,
    )
    return float(maximum)


def design_axial(
    collinear_point: int,
    amplitude_km: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
) -> Orbit:
    """生成指定面外振幅的 Axial 周期轨道（Gómez Type B 分岔族）。

    ``amplitude_km`` 带符号：正为上族、负为下族。振幅 = 一个周期内
    ``|z|`` 的最大值（km）。以面外速度 ``vz0`` 为族参数沿 Type B 分支行
    走，从 Lyapunov 分岔邻域的小振幅种子出发逼近目标。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    du = dynamics.system.characteristic_length
    assert du is not None
    z_target = abs(amplitude_km) / du
    sign = 1.0 if amplitude_km >= 0 else -1.0

    return _walk_family(
        correct_at=lambda vz0, guess: _correct_axial(dynamics, vz0, collinear_point, guess),
        measure=lambda orbit: _z_amplitude_max(dynamics, orbit),
        target=z_target,
        p_seed=float(np.copysign(_AXIAL_SEED_VZ0, sign)),
        dp_init=float(np.copysign(0.002, sign)),
        max_step=0.004,
        tol=1e-6,
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
    corrector.setup_spo_fixed_x0(x0=x0, libration_point=libration_point)
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
        corrector.setup_lpo_fixed_x0(x0=x0, libration_point=libration_point)
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


# ---------------------------------------------------------------------------
# Facade 轨道族生成适配器。八族数值生成均经 generate_rust_family 单次调用；
# 本节只保留参数守卫、领域分派和兼容返回投影。
# ---------------------------------------------------------------------------


def design_dro_family(
    min_amplitude_km: float,
    max_amplitude_km: float,
    *,
    n_orbits: int = 50,
    dynamics: CR3BP_Dynamics | None = None,
) -> FamilyGenerationResult:
    """生成 DRO 族：月心逆行族中振幅落入请求范围的成员。

    振幅定义同 ``design_dro``（一个周期内距月心距离 min/max 均值，km）。
    DRO 不绑定平动点；族参数为近侧 x 轴穿越点 ``x0``，从标准种子出发
    单次自然参数延拓（修正失败步长减半），按请求窗口与种子振幅的相对
    位置选择行走方向（跨种子窗口双向行走），收集振幅落入
    ``[min_amplitude_km, max_amplitude_km]`` 的成员，至多 ``n_orbits`` 条，
    按振幅升序排列。

    Args:
        min_amplitude_km: 族振幅下限（km）。
        max_amplitude_km: 族振幅上限（km）。
        n_orbits: 族成员数量上限。
        dynamics: CR3BP 动力学；缺省构造标准地月系统。

    Returns:
        :class:`FamilyGenerationResult`；``family`` 是 DRO 成员组成的
        ``OrbitFamily``（``family_type="dro"``），软失败时保留部分成员。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    if n_orbits < 1:
        raise ValueError(f"n_orbits 必须大于 0，当前为 {n_orbits}")
    if min_amplitude_km <= 0.0 or min_amplitude_km >= max_amplitude_km:
        raise ValueError("振幅范围必须满足 0 < min_amplitude_km < max_amplitude_km")
    from .rust_generation import generate_rust_family

    return generate_rust_family(
        "dro",
        0,
        n_orbits,
        dynamics,
        min_amplitude_km=min_amplitude_km,
        max_amplitude_km=max_amplitude_km,
    )


def design_nrho_family(
    libration_point: int,
    north_south: int,
    perilune_height_max_km: float,
    *,
    n_orbits: int = 50,
    continuation_direction: str = "toward-moon",
    dynamics: CR3BP_Dynamics | None = None,
) -> FamilyGenerationResult:
    """生成 NRHO 族：Halo 族折叠点后近月段中近月点高度达标的成员。

    NRHO 不是独立的族，而是 Halo 族越过折叠点后的近直线段；族成员 =
    近月点高度 ≤ ``perilune_height_max_km`` 的连续段，至多 ``n_orbits``
    条。Rust 的 L1 路径从小振幅 Halo 种子做单次 PAL，L2 从 DE421 地月
    模型标定的折叠后成员固定 x0 向月侧延拓；每个成员在加入结果前重新
    修正并测量近月点。

    Args:
        libration_point: 平动点编号（1=L1, 2=L2）。
        north_south: 1=北族，2=南族。
        perilune_height_max_km: 族成员的近月点高度上限（km）。
        n_orbits: 族成员数量上限。
        dynamics: CR3BP 动力学；缺省构造标准地月系统。

    Returns:
        :class:`FamilyGenerationResult`；``family`` 是近月段成员组成的
        ``OrbitFamily``（``family_type="nrho"``），软失败时保留部分成员。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    if libration_point not in (1, 2):
        raise ValueError(f"libration_point 必须为 1 或 2，当前为 {libration_point}")
    if north_south not in (1, 2):
        raise ValueError(f"north_south 必须为 1 或 2，当前为 {north_south}")
    if n_orbits < 1:
        raise ValueError(f"n_orbits 必须大于 0，当前为 {n_orbits}")
    if perilune_height_max_km <= 0.0:
        raise ValueError("perilune_height_max_km 必须为正数")
    if continuation_direction != "toward-moon":
        raise ValueError("NRHO continuation_direction 仅支持 'toward-moon'")
    from .rust_generation import generate_rust_family

    return generate_rust_family(
        "nrho",
        libration_point,
        n_orbits,
        dynamics,
        north_south=north_south,
        perilune_height_max_km=perilune_height_max_km,
    )


def design_axial_family(
    libration_point: int,
    max_amplitude_km: float,
    *,
    n_orbits: int = 50,
    continuation_direction: str = "increase-amplitude",
    dynamics: CR3BP_Dynamics | None = None,
) -> FamilyGenerationResult:
    """生成 Axial 族（Gómez Type B 分岔族）成员。

    以面外速度 ``vz0`` 为族参数，从 Lyapunov 分岔邻域的小振幅成员出
    发等步行走（失败步长减半），收集 ``|z|`` 振幅不超过
    ``|max_amplitude_km|`` 的成员，至多 ``n_orbits`` 条。
    ``max_amplitude_km`` 带符号：正为上族、负为下族（与
    ``design_axial`` 约定一致）。

    Args:
        libration_point: 平动点编号（1=L1, 2=L2）。
        max_amplitude_km: 族振幅上限（km），带符号区分上/下族。
        n_orbits: 族成员数量上限。
        dynamics: CR3BP 动力学；缺省构造标准地月系统。

    Returns:
        :class:`FamilyGenerationResult`；``family`` 是 Axial 成员组成的
        ``OrbitFamily``（``family_type="axial"``）。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    if libration_point not in (1, 2):
        raise ValueError(f"libration_point 必须为 1 或 2，当前为 {libration_point}")
    if n_orbits < 1:
        raise ValueError(f"n_orbits 必须大于 0，当前为 {n_orbits}")
    if max_amplitude_km == 0.0:
        raise ValueError("max_amplitude_km 不能为 0")
    if continuation_direction != "increase-amplitude":
        raise ValueError("Axial continuation_direction 仅支持 'increase-amplitude'")
    from .rust_generation import generate_rust_family

    return generate_rust_family(
        "axial",
        libration_point,
        n_orbits,
        dynamics,
        max_amplitude_km=max_amplitude_km,
    )


def design_lissajous_family(
    libration_point: int,
    amplitude_in_km: float,
    amplitude_out_km: float,
    phase_in: float,
    phase_out: float,
    *,
    n_orbits: int = 50,
    sampling_mode: str = "linear-amplitudes",
    dynamics: CR3BP_Dynamics | None = None,
    n_periods: int = 3,
) -> FamilyGenerationResult:
    """生成 Lissajous 拟周期轨迹采样族。

    Lissajous 面内/面外频率不可约，不是周期族；族生成是参数采样而
    非延拓：面内/面外振幅从请求值的 1/n 到 1 线性插值取
    ``n_orbits`` 个样本，两相位固定。每个成员是 Rust 非线性中心约化流上的
    有界多点轨迹（语义同 ``design_lissajous``），无周期闭合，不得
    按严格周期族消费——族上显式标注
    ``metadata["periodicity"] = "quasi-periodic"``。

    Args:
        libration_point: 共线点编号（1/2/3）。
        amplitude_in_km / amplitude_out_km: 面内/面外振幅上限（km）。
        phase_in / phase_out: 面内/面外初始相位（0~1，采样中固定）。
        n_orbits: 采样成员数。
        dynamics: CR3BP 动力学；缺省构造标准地月系统。
        n_periods: 每条轨迹覆盖的名义周期数（默认 3）。

    Returns:
        :class:`FamilyGenerationResult`；``family`` 是拟周期成员组成的
        ``OrbitFamily``（``family_type="lissajous"``，
        ``is_quasi_periodic`` 为真）。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    if libration_point not in (1, 2, 3):
        raise ValueError(f"libration_point 必须为 1、2 或 3，当前为 {libration_point}")
    if n_orbits < 1:
        raise ValueError(f"n_orbits 必须大于 0，当前为 {n_orbits}")
    if n_periods < 1:
        raise ValueError(f"n_periods 必须大于 0，当前为 {n_periods}")
    if amplitude_in_km <= 0.0 or amplitude_out_km <= 0.0:
        raise ValueError("Lissajous 振幅必须为正数")
    if not 0.0 <= phase_in <= 1.0 or not 0.0 <= phase_out <= 1.0:
        raise ValueError("Lissajous 相位必须在 [0, 1] 内")
    if sampling_mode != "linear-amplitudes":
        raise ValueError("Lissajous sampling_mode 仅支持 'linear-amplitudes'")
    from .rust_generation import generate_rust_family

    return generate_rust_family(
        "lissajous",
        libration_point,
        n_orbits,
        dynamics,
        amplitude_in_km=amplitude_in_km,
        amplitude_out_km=amplitude_out_km,
        phase_in=phase_in,
        phase_out=phase_out,
        n_periods=n_periods,
    )


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
    if min_amplitude_km <= 0.0 or min_amplitude_km >= max_amplitude_km:
        raise ValueError("振幅范围必须满足 0 < min_amplitude_km < max_amplitude_km")
    if continuation_direction not in ("decrease-x0", "increase-x0"):
        raise ValueError("三角族 continuation_direction 必须为 'decrease-x0' 或 'increase-x0'")
    if match_tolerance_km <= 0.0:
        raise ValueError("match_tolerance_km 必须为正数")
    from .rust_generation import generate_rust_family

    return generate_rust_family(
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
