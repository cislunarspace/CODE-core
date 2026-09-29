"""族行走与共享助手：错误类型、地月系统构造、测量与 rust 族生成。

原 ``cr3bp_orbits.py`` 的公共骨架：``Cr3bpOrbitError`` 携带统一状态
三元组；``earth_moon_system`` 构造标准地月 CR3BP 系统；``_walk_family``
用割线思想沿族行走使命中目标标量（振幅/近月距等）；度量助手
（距月/距地 min-max、|z|/|y| 振幅）经 ``orbit_family_metric_py`` 统一
测量，族行走与测试断言共用同一路径。``_generate_rust_family`` 统一
``rust_generation`` 的缺省 dynamics 构造与懒加载调用。
"""

from __future__ import annotations

from collections.abc import Callable

from ....data.templates.seed import CHAR_LENGTH_KM, CHAR_PERIOD_SEC, EARTH_MOON_MU
from ....data.types.orbit import Orbit
from ....integrators import orbit_family_metric_py
from ....status import ConvergenceState, FailureCause, ResultStatus
from ...dynamics import CR3BP_Dynamics, CR3BP_System
from ...solver.differential_correction import DifferentialCorrection


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


def _earth_distance_minmax(
    dynamics: CR3BP_Dynamics, orbit: Orbit, n_points: int = 4000
) -> tuple[float, float]:
    """传播一个周期，返回距地心距离的最小/最大值（无量纲）。

    与 ``_moon_distance_minmax`` 同一测量路径（4000 点采样压制近点
    采样噪声），族行走与测试断言共用。
    """
    assert orbit.period is not None  # 周期轨道必有 period
    minimum, maximum = orbit_family_metric_py(
        float(dynamics.system.mu),
        "earth-distance",
        0,
        orbit.states[0],
        float(orbit.period),
        sample_count=n_points,
    )
    return float(minimum), float(maximum)


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


def _y_amplitude_max(dynamics: CR3BP_Dynamics, orbit: Orbit, n_points: int = 1000) -> float:
    """传播一个周期，返回 |y| 的最大值（无量纲 DU）。"""
    assert orbit.period is not None
    _, maximum = orbit_family_metric_py(
        float(dynamics.system.mu),
        "y-amplitude",
        0,
        orbit.states[0],
        float(orbit.period),
        sample_count=n_points,
    )
    return float(maximum)


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


def _generate_rust_family(family_type, libration_point, n_orbits, dynamics, **parameters):
    """统一 rust_generation 调用：dynamics 缺省构造 + 函数局部导入（保持懒加载，不引入环）。"""
    if dynamics is None:
        from ...dynamics import CR3BP_Dynamics

        dynamics = CR3BP_Dynamics(earth_moon_system())
    from ..rust_generation import generate_rust_family

    return generate_rust_family(family_type, libration_point, n_orbits, dynamics, **parameters)


def _validate_amplitude_range(min_amplitude_km: float, max_amplitude_km: float) -> None:
    if not (0.0 < min_amplitude_km < max_amplitude_km):
        raise ValueError("振幅范围必须满足 0 < min_amplitude_km < max_amplitude_km")
