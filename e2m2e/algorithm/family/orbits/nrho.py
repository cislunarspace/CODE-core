"""NRHO 族：Halo 族折叠点后的近月段特选成员。

按近月距（距月心 = 近月点高度 + 月球半径）选取，北/南对应
``halo_class`` 0/1。L2：固定 z0 走到折叠点前的族成员，再固定 x0 向
月侧行走；L1：固定 x0 在折叠点两侧均失效，改用 PAL 延拓。
"""

from __future__ import annotations

import numpy as np

from ....data.templates.seed import _HALO_SEED_Z0, MOON_RADIUS_KM
from ....data.types.orbit import Orbit
from ...dynamics import CR3BP_Dynamics
from ...results import FamilyGenerationResult
from ...solver.differential_correction import DifferentialCorrection
from ..halo_family import halo_pseudo_arclength_continuation
from .halo import _correct_halo, _correct_halo_x0, _halo_seed_walk
from .walk import (
    Cr3bpOrbitError,
    _generate_rust_family,
    _moon_distance_minmax,
    _require_orbit,
    _walk_family,
    earth_moon_system,
)


def _pal_termination_reason(perilunes: list[float], max_orbits: int, target_du: float) -> str:
    """PAL 未达目标即耗尽时的终止原因消息。

    区分三种失败：族已到尽头（成员近月距单调演化，家族在步数内未提供
    更低近月段）、步进停滞（尾部成员近月距往复振荡，延拓没有沿族推进，
    即 #772 修复前的双能级 2-周期振荡形态）与提前中断（成员数少于请求
    条数，内层微分修正失败 break）。停滞判据取尾部至多 50 条成员近月距
    序列的方向反转次数：单调演化反转 ≤1 次，2-周期振荡约每两步反转一次，
    阈值 8 居中。
    """
    if not perilunes:
        return f"PAL 延拓未产生任何收敛成员，未能到达目标近月距 {target_du:.6f} DU"
    tail = perilunes[-50:]
    reversals = 0
    prev_dir = 0
    for cur, nxt in zip(tail, tail[1:], strict=False):
        delta = nxt - cur
        if abs(delta) < 1e-12:
            continue
        cur_dir = 1 if delta > 0 else -1
        if prev_dir and cur_dir != prev_dir:
            reversals += 1
        prev_dir = cur_dir
    if len(tail) >= 8 and reversals >= 8:
        return (
            f"PAL 延拓步进停滞：尾部 {len(tail)} 条成员近月距往复振荡 "
            f"（{reversals} 次方向反转），未能到达目标近月距 {target_du:.6f} DU"
        )
    if len(perilunes) < max_orbits:
        # PAL 未跑满请求条数即中断（内层微分修正失败 break），此时既不是
        # 族已到尽头，也不该按请求条数报账
        return (
            f"PAL 延拓在 {len(perilunes)} 条成员后提前中断（未达请求的 {max_orbits} 条），"
            f"最小近月距 {min(perilunes):.6f} DU 仍未到达目标近月距 {target_du:.6f} DU"
        )
    return (
        f"PAL 延拓 {max_orbits} 条轨道后族已到尽头，最小近月距 "
        f"{min(perilunes):.6f} DU 仍未到达目标近月距 {target_du:.6f} DU"
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
    单次调用则可一路走到近月距 0.003 DU 以下。北/南半球都走 positive
    振幅增长支（``halo_class`` 决定目标方向 td），未达目标抛出的
    ``Cr3bpOrbitError`` 区分族已到尽头、步进停滞与提前中断三种原因。
    """
    from ...solver.continuation import Continuation

    seed = _correct_halo(dynamics, float(np.copysign(_HALO_SEED_Z0, z_sign)), libration_point, None)
    seed.family_type = "halo"
    seed.parameters = {
        "libration_point": libration_point,
        "halo_class": 0 if z_sign > 0 else 1,
        "amplitude_z": _HALO_SEED_Z0,
    }
    continuation = Continuation(corrector=DifferentialCorrection(dynamics))

    # 北/南半球都走振幅增长支：positive 支按 halo_class 取目标方向（北
    # td=+1、南 td=-1），两半球都从种子向折叠点爬升；negative 支以 x 为
    # 目标、走向平面分岔端，到不了近月段（#772：南族此前误走 negative 支）。
    family = halo_pseudo_arclength_continuation(
        continuation,
        seed_orbit=seed,
        n_orbits=max_orbits,
        direction="positive",
        step_size=0.0045,
        verbose=False,
    )
    prev_orbit = seed
    perilunes: list[float] = []
    for orb in list(family.orbits)[1:]:
        perilune = _moon_distance_minmax(dynamics, orb)[0]
        perilunes.append(perilune)
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

    raise Cr3bpOrbitError(_pal_termination_reason(perilunes, max_orbits, target_du))


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
    return _generate_rust_family(
        "nrho",
        libration_point,
        n_orbits,
        dynamics,
        north_south=north_south,
        perilune_height_max_km=perilune_height_max_km,
    )
