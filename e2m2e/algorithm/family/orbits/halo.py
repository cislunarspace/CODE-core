"""Halo 族：共线平动点面外周期轨道。

Richardson 三阶近似种子 + 定 ``z0`` 微分修正，沿族把 ``z0`` 走到
目标面外振幅（km，符号区分北/南）。``|z0|`` 在固定 z0 延拓安全上界内
直接行走；L2 超界改用固定 ``x0`` 行走，L1 超界报错（族越过折叠点后
转入近月 NRHO 段，经 ``design_nrho`` / ``design_nrho_family`` 获取）。
"""

from __future__ import annotations

import numpy as np

from ....data.templates.seed import _HALO_FIXED_Z0_LIMIT, _HALO_SEED_Z0
from ....data.types.orbit import Orbit, OrbitFamily
from ....status import ConvergenceState
from ...dynamics import CR3BP_Dynamics
from ...results import FamilyGenerationResult
from ...solver.differential_correction import DifferentialCorrection
from ..halo_initial_guess import compute_halo_initial_guess
from ..strategies import halo_fixed_x0, halo_fixed_z0
from .walk import (
    Cr3bpOrbitError,
    _correct_or_raise,
    _generate_rust_family,
    _require_orbit,
    _walk_family,
    earth_moon_system,
)

#: L1 Halo 族折叠点实测无量纲 |z0|（PAL 越折测试，×特征长度 ≈32674 km）；
#: 固定 z0 延拓安全上界 ``_HALO_FIXED_Z0_LIMIT[1]``（0.07）是其前的保守
#: 截断，二者不是一回事（#773）
_L1_FOLD_Z0 = 0.085


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
    corrector.configure(halo_fixed_z0(z0=z0, libration_point=libration_point))
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

    固定 ``z0`` 的修正在 Halo 族折叠点前失效（实测折叠点 L1 约
    ``|z0|≈0.085``、L2 约 ``|z0|≈0.20``；固定 z0 行走的安全上界
    ``_HALO_FIXED_Z0_LIMIT`` 取其前的保守截断值）；折叠前后 ``x0``
    单调，改用它作族参数可一路走到 NRHO。
    """
    state = guess.states[0].copy()
    state[0] = x0
    corrector = DifferentialCorrection(dynamics)
    corrector.configure(halo_fixed_x0(x0=x0, libration_point=libration_point))
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = guess.period
    return _correct_or_raise(corrector, seed, f"Halo(L{libration_point}, x0={x0:.6f})")


def _halo_seed_walk(
    dynamics: CR3BP_Dynamics,
    libration_point: int,
    z_sign: float,
) -> Orbit:
    """固定 ``z0`` 从 Richardson 种子走到安全上界 ``_HALO_FIXED_Z0_LIMIT``，
    作为固定 ``x0`` 行走的出发成员。"""
    return _walk_family(
        correct_at=lambda z0, guess: _correct_halo(dynamics, z0, libration_point, guess),
        measure=lambda orbit: float(orbit.states[0, 2]),
        target=float(np.copysign(_HALO_FIXED_Z0_LIMIT[libration_point], z_sign)),
        p_seed=float(np.copysign(_HALO_SEED_Z0, z_sign)),
        dp_init=float(np.copysign(0.01, z_sign)),
        max_step=0.02,
        tol=1e-6,
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
    ``|z0|`` 不超过固定 z0 延拓安全上界 ``_HALO_FIXED_Z0_LIMIT``（折叠点
    前的保守截断）时直接固定 z0 行走；L2 更大振幅先走到安全上界处的族
    成员，再改用固定 x0 行走逼近目标；L1 超界抛 ``Cr3bpOrbitError``
    （族越过折叠点后转入近月 NRHO 段，改经 ``design_nrho`` /
    ``design_nrho_family`` 获取）。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    du = dynamics.system.characteristic_length
    assert du is not None
    z_target = amplitude_km / du

    if abs(z_target) <= _HALO_FIXED_Z0_LIMIT[collinear_point]:
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
        # L1 族越过折叠点（|z0|≈_L1_FOLD_Z0）后 z0 回落、族转入近月 NRHO 段
        # 继续延伸：目标振幅的成员存在，只是固定 z0 / 固定 x0 延拓都到不了
        # （固定 x0 在 L1 会误入平面/垂直 Lyapunov 族）；明确报错并指向
        # NRHO 设计路径，而不是返回错误的族
        raise Cr3bpOrbitError(
            f"L1 Halo 固定 z0 延拓的安全上界约 {_HALO_FIXED_Z0_LIMIT[1] * du:.0f} km"
            f"（无量纲 |z0|≤{_HALO_FIXED_Z0_LIMIT[1]}，折叠点前的保守截断），"
            f"目标 {amplitude_km:.0f} km 超出本路径能力；族折叠点约 "
            f"{_L1_FOLD_Z0 * du:.0f} km（|z0|≈{_L1_FOLD_Z0}），越过折叠点后族转入"
            f"近月 NRHO 段继续延伸，更大面外振幅的成员可经 design_nrho / "
            f"design_nrho_family 获得"
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
    至多 ``n_orbits`` 条（含种子）。振幅上限不得超过该平动点固定 z0 延拓
    的安全上界（``seed._HALO_FIXED_Z0_LIMIT``，族折叠点前的保守截断）：
    安全上界之后固定 z0 延拓失效，族越过折叠点后转入近月 NRHO 段（经
    ``design_nrho_family`` 获取）。

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
    result = _generate_rust_family(
        "halo",
        libration_point,
        n_orbits,
        dynamics,
        max_amplitude_km=max_amplitude_km,
    )
    if result.status is ConvergenceState.CONVERGED:
        return result.family
    return result
