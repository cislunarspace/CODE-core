"""RO 恒星共振族：绕地质心的顺行平面周期轨道。

p:q = 航天器惯性圈数:月球圈数，闭合条件为 q 个恒星月内绕地 p 圈：
会合系周期 ``T = 2πq``、净卷绕 ``w = p − q``。精确成员从文献偏心族
入口上族并割线钉定目标周期；指定振幅时以 +x 轴穿越点 ``x0`` 为族参数
沿族行走命中目标。振幅定义：一个周期内距地心距离 min/max 均值（km）。
"""

from __future__ import annotations

import numpy as np

from ....data.templates import RO_SUPPORTED_RESONANCES
from ....data.types.orbit import Orbit
from ...dynamics import CR3BP_Dynamics
from ...results import FamilyGenerationResult
from ...solver.differential_correction import DifferentialCorrection
from ..strategies import symmetric_2d_fixed_x0
from .walk import (
    Cr3bpOrbitError,
    _correct_or_raise,
    _earth_distance_minmax,
    _generate_rust_family,
    _require_orbit,
    _validate_amplitude_range,
    _walk_family,
    earth_moon_system,
)


def _ro_eccentric_seed(
    dynamics: CR3BP_Dynamics,
    p: int,
    q: int,
    e: float,
    apsis: str,
) -> tuple[np.ndarray, float, float]:
    """构造恒星 p:q 共振的偏心 Kepler 种子（状态、周期、半长轴）。

    p:q = 航天器惯性圈数:月球圈数（Vaquero & Howell 2014 式（7）、
    （13）），闭合条件为 q 个恒星月内绕地 p 圈：会合系周期
    ``T = 2πq``、净卷绕 ``w = p − q``。``apsis`` 为 ``"peri"`` 时取
    +x 近心点，为 ``"apo"`` 时取 +x 远心点。状态速度先按地心二体
    问题计算，再减去会合系旋转速度；远心点路径采用上族的正 ``vy0``
    符号。
    """
    if not 0.0 <= e < 1.0:
        raise ValueError(f"偏心率必须满足 0 <= e < 1，当前为 {e}")
    if apsis not in ("peri", "apo"):
        raise ValueError(f"apsis 必须为 peri 或 apo，当前为 {apsis!r}")
    mu = float(dynamics.system.mu)
    a = ((1.0 - mu) * (q / p) ** 2) ** (1.0 / 3.0)
    r = a * (1.0 - e if apsis == "peri" else 1.0 + e)
    x0 = r - mu
    inertial_speed = float(np.sqrt((1.0 - mu) * (2.0 / r - 1.0 / a)))
    vy0 = inertial_speed - x0 if apsis == "peri" else x0 - inertial_speed
    period = 2.0 * np.pi * q
    return np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0]), period, a


def _guard_ro_branch(
    dynamics: CR3BP_Dynamics,
    p: int,
    q: int,
    orbit: Orbit,
    a_kepler: float,
    *,
    n_points: int = 100,
) -> Orbit:
    """拒绝伪支：振幅量级、偏心形态与卷绕数都必须落在目标族上。

    目录五族均为偏心轨道（r_max/r_min ≥ 1.2）：近圆 w=1/w=3 诱饵支
    由形态闸拒绝，微型自旋与跨支伪解由卷绕闸（w = p−q）拒绝。卷绕
    从修正轨迹的采样直接测量，不额外传播。
    """
    d_min, d_max = _earth_distance_minmax(dynamics, orbit, n_points=n_points)
    amplitude = 0.5 * (d_min + d_max)
    if abs(amplitude - a_kepler) / a_kepler > 0.1:
        raise Cr3bpOrbitError(
            f"RO({p}:{q}) 跳到错误振幅支：均值={amplitude:.6f} DU、a={a_kepler:.6f} DU"
        )
    if d_max / d_min < 1.2:
        raise Cr3bpOrbitError(f"RO({p}:{q}) 跳到近圆伪支：r_max/r_min={d_max / d_min:.3f} < 1.2")
    theta = np.unwrap(np.arctan2(orbit.states[:, 1], orbit.states[:, 0]))
    winding = float((theta[-1] - theta[0]) / (2.0 * np.pi))
    if abs(winding - (p - q)) > 0.01:
        raise Cr3bpOrbitError(f"RO({p}:{q}) 卷绕数 {winding:+.4f} 不等于目标 {p - q:+d}")
    return orbit


def _validate_ro_exact(
    dynamics: CR3BP_Dynamics,
    p: int,
    q: int,
    orbit: Orbit,
) -> Orbit:
    """最终验收：闭合契约 + 统一伪支守卫。"""
    if orbit.closure_error is None or orbit.closure_error >= 1e-6:
        raise Cr3bpOrbitError(f"RO({p}:{q}) 精确成员闭合误差超限：{orbit.closure_error}")
    # 割线钉定的最终闭合容差为 1e-9·T，可能略宽于 Orbit 构造器的
    # 研究级 is_periodic 阈值；这里按 RO 的公开闭合契约统一标记。
    orbit.is_periodic = True
    mu = float(dynamics.system.mu)
    a_kepler = ((1.0 - mu) * (q / p) ** 2) ** (1.0 / 3.0)
    return _guard_ro_branch(dynamics, p, q, orbit, a_kepler, n_points=4000)


def _ro_fast_21_seed(dynamics: CR3BP_Dynamics, period: float, a_kepler: float) -> Orbit:
    """构造 2:1 数值折返点附近的局部偏心种子。

    2:1 偏心族从 e=0.5 近心点种子沿固定 x0 行走会在折返点附近
    需要数百次小步修正。这里仍以同一 Kepler 半长轴和目标周期为约束，
    从折返点附近的局部近心点初猜扫描一个很小的状态窗口，找到同一
    偏心支后再用周期行走钉定；不接受振幅偏离 Kepler 半长轴 10% 的支。
    """
    state, _, _ = _ro_eccentric_seed(dynamics, 2, 1, 0.75, "peri")
    for x_offset in (0.0008, 0.0010, 0.0011, 0.0012, 0.0014):
        for vy_offset in (-0.04, -0.03, -0.025, -0.02, -0.01, 0.0):
            candidate_state = state.copy()
            candidate_state[0] += x_offset
            candidate_state[4] += vy_offset
            candidate = Orbit(
                states=candidate_state.reshape(1, -1),
                times=np.array([0.0]),
                system=dynamics.system,
            )
            candidate.period = period
            try:
                orbit = _correct_ro(dynamics, float(candidate_state[0]), candidate)
            except Cr3bpOrbitError:
                continue
            try:
                return _guard_ro_branch(dynamics, 2, 1, orbit, a_kepler)
            except Cr3bpOrbitError:
                continue
    raise Cr3bpOrbitError("RO(2:1) 局部偏心种子未找到正确振幅支")


def _ro_fast_32_seed(dynamics: CR3BP_Dynamics, period: float, a_kepler: float) -> Orbit:
    """构造 3:2 近心点上族的局部启动成员。"""
    state, _, _ = _ro_eccentric_seed(dynamics, 3, 2, 0.46, "peri")
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = period
    orbit = _correct_ro(dynamics, float(state[0]), seed)
    return _guard_ro_branch(dynamics, 3, 2, orbit, a_kepler)


def _ro_fast_43_seed(dynamics: CR3BP_Dynamics, period: float, a_kepler: float) -> Orbit:
    """构造 4:3 远心点上族的局部启动成员。"""
    state, _, _ = _ro_eccentric_seed(dynamics, 4, 3, 0.76, "apo")
    state[4] = -1.125
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = period
    orbit = _correct_ro(dynamics, float(state[0]), seed)
    return _guard_ro_branch(dynamics, 4, 3, orbit, a_kepler)


#: 3:1/4:1 的偏心 Kepler 种子偏心率：取值使近心点 x0 落在目录精确
#: 成员穿越点的邻域（3:1 x0≈0.021、4:1 x0≈0.280），固定 x0 修正
#: 直接落在目标偏心支上。
_RO_PLAIN_SEED_E: dict[tuple[int, int], float] = {(3, 1): 0.93, (4, 1): 0.26}


def _ro_branch_entry(
    dynamics: CR3BP_Dynamics,
    p: int,
    q: int,
    period: float,
    a_kepler: float,
) -> tuple[Orbit, float]:
    """构造文献偏心族上的入口成员，并给出割线钉定步长。

    2:1/3:2/4:3 的族在数值折返区附近，入口需局部扫描或特调种子
    （``_ro_fast_*_seed``）；3:1/4:1 的偏心 Kepler 种子直接可用。
    步长的符号朝周期减小的一侧取第二点，使目标周期夹在割线区间内。
    """
    if (p, q) == (2, 1):
        return _ro_fast_21_seed(dynamics, period, a_kepler), 0.0002
    if (p, q) == (3, 2):
        return _ro_fast_32_seed(dynamics, period, a_kepler), 0.0002
    if (p, q) == (4, 3):
        return _ro_fast_43_seed(dynamics, period, a_kepler), 0.0003
    eccentricity = _RO_PLAIN_SEED_E[(p, q)]
    state, _, _ = _ro_eccentric_seed(dynamics, p, q, eccentricity, "peri")
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = period
    return _correct_ro(dynamics, float(state[0]), seed), -0.0025


def _ro_correct_trial(
    dynamics: CR3BP_Dynamics, p: int, q: int, x_try: float, guess: Orbit
) -> Orbit:
    """修正族参数 x_try 处的 RO 成员（族行走与割线钉定共用）。

    3:1/4:1 的 x0–vy0 映射在目标邻域很陡，邻点续猜会跳支或不收敛；
    这两档按 x_try 重建近心点 Kepler 种子（逐点收敛已实证），其余三档
    沿用邻点续猜。
    """
    if (p, q) not in _RO_PLAIN_SEED_E:
        return _correct_ro(dynamics, x_try, guess)
    mu = float(dynamics.system.mu)
    a_kepler = ((1.0 - mu) * (q / p) ** 2) ** (1.0 / 3.0)
    eccentricity = 1.0 - (x_try + mu) / a_kepler
    if not 0.0 <= eccentricity < 1.0:
        raise Cr3bpOrbitError(f"RO({p}:{q}) 试探点超出偏心种子域：x={x_try:.6f}")
    state, _, _ = _ro_eccentric_seed(dynamics, p, q, eccentricity, "peri")
    fresh = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    fresh.period = 2.0 * np.pi * q
    return _correct_ro(dynamics, x_try, fresh)


def _ro_pin_period_secant(
    dynamics: CR3BP_Dynamics,
    p: int,
    q: int,
    seed: Orbit,
    a_kepler: float,
    dx: float,
) -> Orbit:
    """在局部偏心上族用割线法钉定目标周期 ``T = 2πq``。"""
    target = 2.0 * np.pi * q
    tolerance = 1e-9 * target
    left = _guard_ro_branch(dynamics, p, q, seed, a_kepler)
    assert left.period is not None
    x_left = float(left.states[0, 0])
    right = _guard_ro_branch(
        dynamics,
        p,
        q,
        _ro_correct_trial(dynamics, p, q, x_left + dx, left),
        a_kepler,
    )
    assert right.period is not None
    x_right = float(right.states[0, 0])
    for _ in range(8):
        denominator = right.period - left.period
        if denominator == 0.0:
            break
        x_try = x_left + (target - left.period) / denominator * (x_right - x_left)
        guess = left if abs(x_try - x_left) <= abs(x_try - x_right) else right
        candidate = _guard_ro_branch(
            dynamics,
            p,
            q,
            _ro_correct_trial(dynamics, p, q, x_try, guess),
            a_kepler,
        )
        assert candidate.period is not None
        if abs(candidate.period - target) <= tolerance:
            return candidate
        if (candidate.period - target) * (left.period - target) <= 0.0:
            right, x_right = candidate, x_try
        else:
            left, x_left = candidate, x_try
    raise Cr3bpOrbitError(f"RO({p}:{q}) 割线周期钉定未命中目标")


def _correct_ro_exact(dynamics: CR3BP_Dynamics, p: int, q: int) -> Orbit:
    """修正恒星 p:q 共振的精确成员（文献偏心族 w = p−q 支）。

    闭合条件为 q 个恒星月内绕地 p 圈：会合系周期 ``T = 2πq``、净
    卷绕 ``w = p − q``（resonant.csv 目录各族的精确成员同此约定，
    与 Vaquero & Howell 2014 的 p:q 定义一致）。构造 = 偏心族入口
    （``_ro_branch_entry``）+ 割线钉定周期；闭合误差、Kepler 半长轴
    量级、偏心形态与卷绕数四重守卫拒绝近圆诱饵与多圈伪解。
    """
    period = 2.0 * np.pi * q
    mu = float(dynamics.system.mu)
    a_kepler = ((1.0 - mu) * (q / p) ** 2) ** (1.0 / 3.0)
    entry, dx = _ro_branch_entry(dynamics, p, q, period, a_kepler)
    orbit = _ro_pin_period_secant(dynamics, p, q, entry, a_kepler, dx=dx)
    return _validate_ro_exact(dynamics, p, q, orbit)


def _correct_ro(dynamics: CR3BP_Dynamics, x0: float, guess: Orbit) -> Orbit:
    """在 +x 轴穿越点 ``x0`` 处修正 RO 族成员（固定 x0，自由 vy0 与半周期）。

    族行走沿共振族离开精确通约点时周期随之漂移；伪解判定对照初猜
    （上一步成员）周期：跳变超过 ±20% 即视为多圈伪解，交由族行走退
    半步重试（RO 族周期随 x0 双向变化，故双侧判定，与 DRO 单侧不同）。
    """
    state = guess.states[0].copy()
    state[0] = x0
    assert guess.period is not None
    period = guess.period
    corrector = DifferentialCorrection(dynamics)
    corrector.configure(symmetric_2d_fixed_x0(x0=x0))
    # 族折返点附近的 STM 修正量可小于默认停滞阈值，但残差仍能继续
    # 降到收敛容差；过早判停会把真实共振支误报为不可达。
    corrector.stagnation_limit = 1e-16
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = period
    orbit = _correct_or_raise(corrector, seed, f"RO(x0={x0:.6f})")
    assert orbit.period is not None
    if abs(orbit.period - period) > 0.2 * period:
        raise Cr3bpOrbitError(
            f"RO(x0={x0:.6f}) 修正跳到异周期伪解（T={orbit.period:.3f}，初猜 {period:.3f}）"
        )
    return orbit


def design_ro(
    p: int,
    q: int,
    amplitude_km: float | None = None,
    *,
    dynamics: CR3BP_Dynamics | None = None,
    tol_km: float = 20.0,
) -> Orbit:
    """生成 p:q 恒星共振轨道（RO）：绕地质心的顺行平面周期轨道。

    p:q = 航天器惯性圈数:月球圈数，满足
    ``p/q = n_sc/n_moon = T_moon/T_sc``（Vaquero & Howell 2014
    式（7）、（13））。闭合条件为 q 个恒星月内绕地 p 圈：会合系周期
    ``T = 2πq``、净卷绕 ``w = p−q``（3:1 即一个恒星月绕地 3 圈，
    图案周期约 27.3 天、惯性周期约 9.1 天）。不指定 ``amplitude_km``
    时返回精确共振成员：从文献偏心族入口上族并割线钉定目标周期，
    闭合/Kepler 量级/偏心形态/卷绕数四重守卫拒绝伪支。指定振幅时以
    +x 轴穿越点 ``x0`` 为族参数，从精确成员出发沿族行走命中目标
    （命中 ``tol_km`` 内即停）；行走离开精确通约点，周期随振幅漂移。

    振幅定义：一个周期内距地心距离最小/最大值的均值（km），与
    ``design_dro`` 的月心距定义同构。

    支持 ``RO_SUPPORTED_RESONANCES`` 五档顺行内共振（2:1/3:1/3:2/4:1/
    4:3）；其余比值的初猜不可靠，明确拒绝。

    References:
        Vaquero & Howell (2014). Design of transfer trajectories between
        resonant orbits in the Earth–Moon restricted problem. Acta
        Astronautica 94(1).
    """
    if (p, q) not in RO_SUPPORTED_RESONANCES:
        raise ValueError(
            f"不支持的共振比 {p}:{q}；RO 支持 "
            f"{'/'.join(f'{pp}:{qq}' for pp, qq in sorted(RO_SUPPORTED_RESONANCES))}"
            "（顺行内共振，p:q = 卫星:月球）"
        )
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    seed = _correct_ro_exact(dynamics, p, q)
    if amplitude_km is None:
        return seed
    du = dynamics.system.characteristic_length
    assert du is not None

    def measure(orbit: Orbit) -> float:
        d_min, d_max = _earth_distance_minmax(dynamics, orbit)
        return 0.5 * (d_min + d_max)

    return _walk_family(
        correct_at=lambda x0, guess: _ro_correct_trial(dynamics, p, q, x0, _require_orbit(guess)),
        measure=measure,
        target=amplitude_km / du,
        p_seed=float(seed.states[0, 0]),
        dp_init=0.001,
        max_step=0.0025,
        tol=tol_km / du,
        seed_orbit=seed,
    )


def design_ro_family(
    p: int,
    q: int,
    min_amplitude_km: float,
    max_amplitude_km: float,
    *,
    n_orbits: int = 50,
    dynamics: CR3BP_Dynamics | None = None,
) -> FamilyGenerationResult:
    """生成恒星 p:q RO 共振族中振幅落入请求范围的成员。

    p:q = 航天器惯性圈数:月球圈数，精确成员的会合系周期为
    ``T = 2πq``（q 个恒星月）、净卷绕 ``w = p−q``。振幅定义同
    ``design_ro``（一个周期内距地心距离 min/max 均值，km）。RO 不
    绑定平动点，五档均锚定文献偏心族。族参数为 +x 轴穿越点 ``x0``，
    单次自然参数延拓双向行走（修正失败步长减半），收集振幅落入
    ``[min_amplitude_km, max_amplitude_km]`` 的成员，至多 ``n_orbits``
    条，按振幅升序排列。行走离开精确通约点，成员周期随振幅漂移。

    Args:
        p: 共振比卫星侧整数（p:q = 卫星:月球，支持集见
            ``RO_SUPPORTED_RESONANCES``）。
        q: 共振比月球侧整数。
        min_amplitude_km: 族振幅下限（km）。
        max_amplitude_km: 族振幅上限（km）。
        n_orbits: 族成员数量上限。
        dynamics: CR3BP 动力学；缺省构造标准地月系统。

    Returns:
        :class:`FamilyGenerationResult`；``family`` 是 RO 成员组成的
        ``OrbitFamily``（``family_type="ro"``），软失败时保留部分成员。
    """
    if (p, q) not in RO_SUPPORTED_RESONANCES:
        raise ValueError(
            f"不支持的共振比 {p}:{q}；RO 支持 "
            f"{'/'.join(f'{pp}:{qq}' for pp, qq in sorted(RO_SUPPORTED_RESONANCES))}"
            "（顺行内共振，p:q = 卫星:月球）"
        )
    if n_orbits < 1:
        raise ValueError(f"n_orbits 必须大于 0，当前为 {n_orbits}")
    _validate_amplitude_range(min_amplitude_km, max_amplitude_km)
    return _generate_rust_family(
        "ro",
        0,
        n_orbits,
        dynamics,
        resonance_p=p,
        resonance_q=q,
        min_amplitude_km=min_amplitude_km,
        max_amplitude_km=max_amplitude_km,
    )
