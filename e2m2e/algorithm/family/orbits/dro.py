"""DRO/DPO 族：月心平面周期轨道（逆行/顺行对称族）。

DRO 以近侧 x 轴穿越点 ``x0`` 为族参数，从标准种子出发沿族行走，
选取振幅（一个周期内距月心距离 min/max 均值，km）命中目标的成员；
DPO 是与 DRO 对称的顺行族（vy0 < 0），振幅定义同 DRO。
"""

from __future__ import annotations

import numpy as np

from ....data.templates.seed import (
    _DPO_SEED_PERIOD,
    _DPO_SEED_VY0,
    _DPO_SEED_X0,
    _DRO_SEED_PERIOD,
    _DRO_SEED_VY0,
    _DRO_SEED_X0,
)
from ....data.types.orbit import Orbit
from ...dynamics import CR3BP_Dynamics
from ...results import FamilyGenerationResult
from ...solver.differential_correction import DifferentialCorrection
from ..strategies import symmetric_2d_fixed_x0
from .walk import (
    Cr3bpOrbitError,
    _correct_or_raise,
    _generate_rust_family,
    _moon_distance_minmax,
    _validate_amplitude_range,
    _walk_family,
    earth_moon_system,
)


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
    corrector.configure(symmetric_2d_fixed_x0(x0=x0))
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

    DPO 与 DRO 使用相同修正策略（``symmetric_2d_fixed_x0``），
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
    corrector.configure(symmetric_2d_fixed_x0(x0=x0))
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


def design_dro(
    amplitude_km: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
    tol_km: float = 20.0,
) -> Orbit:
    """生成指定振幅的 DRO 周期轨道。

    振幅定义：一个周期内距月心距离最小/最大值的均值（km）。小振幅成员
    接近圆（amplitude=10000 的成员距月范围约 9973~10007 km），振幅越大
    扁平度越高（种子成员约 75328~106244 km）。以近侧 x 轴穿越点 ``x0``
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
    if n_orbits < 1:
        raise ValueError(f"n_orbits 必须大于 0，当前为 {n_orbits}")
    _validate_amplitude_range(min_amplitude_km, max_amplitude_km)
    return _generate_rust_family(
        "dro",
        0,
        n_orbits,
        dynamics,
        min_amplitude_km=min_amplitude_km,
        max_amplitude_km=max_amplitude_km,
    )
