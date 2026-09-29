"""Axial 族：Gómez Type B 分岔族。

以面外速度 ``vz0`` 为族参数，从 xy 平面出发的 3D 周期轨道
（x 轴对称），振幅 = 一个周期内 ``|z|`` 的最大值（km）。
"""

from __future__ import annotations

import numpy as np

from ....data.templates.seed import _AXIAL_SEED_VZ0
from ....data.types.orbit import Orbit
from ...dynamics import CR3BP_Dynamics
from ...results import FamilyGenerationResult
from ...solver.differential_correction import DifferentialCorrection
from ..axial_initial_guess import compute_axial_initial_guess
from ..strategies import axial_fixed_vz0
from .walk import (
    Cr3bpOrbitError,
    _correct_or_raise,
    _generate_rust_family,
    _walk_family,
    _z_amplitude_max,
    earth_moon_system,
)


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
    corrector.configure(axial_fixed_vz0(vz0=vz0, libration_point=libration_point))
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
    if libration_point not in (1, 2):
        raise ValueError(f"libration_point 必须为 1 或 2，当前为 {libration_point}")
    if n_orbits < 1:
        raise ValueError(f"n_orbits 必须大于 0，当前为 {n_orbits}")
    if max_amplitude_km == 0.0:
        raise ValueError("max_amplitude_km 不能为 0")
    if continuation_direction != "increase-amplitude":
        raise ValueError("Axial continuation_direction 仅支持 'increase-amplitude'")
    return _generate_rust_family(
        "axial",
        libration_point,
        n_orbits,
        dynamics,
        max_amplitude_km=max_amplitude_km,
    )
