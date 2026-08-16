"""平面三角平动点周期族的 Rust 全周期 PAL 适配器。"""

from __future__ import annotations

import numpy as np

from ...data.types.orbit import Orbit, OrbitFamily
from ...integrators import planar_full_period_pal_py
from ..dynamics import CR3BP_Dynamics
from ..results import ContinuationResult

_VALID_FAMILY_TYPES = {"spo", "lpo"}
_VALID_DIRECTIONS = {"increase-x0", "decrease-x0"}


def generate_planar_periodic_family(
    dynamics: CR3BP_Dynamics,
    seed_orbit: Orbit,
    *,
    family_type: str,
    libration_point: int,
    n_orbits: int,
    step_size: float,
    initial_direction: str,
) -> ContinuationResult:
    """从平面 SPO/LPO 种子生成完整周期 PAL 轨道族。

    Rust 内核负责传播、STM、相位规范和 Newton 迭代；本函数仅构造问题并将
    原始成员解释为数据层轨道。Horseshoe 是 LPO 的大振幅成员，复用 ``lpo``。
    """
    if family_type not in _VALID_FAMILY_TYPES:
        raise ValueError(
            f"family_type 必须为 {sorted(_VALID_FAMILY_TYPES)}，当前为 {family_type!r}"
        )
    if libration_point not in (4, 5):
        raise ValueError(f"libration_point 必须为 4 或 5，当前为 {libration_point}")
    if n_orbits < 1:
        raise ValueError(f"n_orbits 必须大于零，当前为 {n_orbits}")
    if step_size <= 0.0:
        raise ValueError(f"step_size 必须为正数，当前为 {step_size}")
    if initial_direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"initial_direction 必须为 {sorted(_VALID_DIRECTIONS)}，当前为 {initial_direction!r}"
        )
    if seed_orbit.period is None:
        raise ValueError("seed_orbit 必须设置完整周期")
    if not np.allclose(seed_orbit.states[0, (2, 5)], 0.0, atol=1e-12):
        raise ValueError("平面全周期 PAL 仅接受 z=vz=0 的种子")

    raw = planar_full_period_pal_py(
        dynamics.system.mu,
        seed_orbit.states[0].tolist(),
        float(seed_orbit.period),
        n_orbits,
        step_size,
        initial_direction,
        rtol=dynamics.rtol,
        atol=dynamics.atol,
        max_step=dynamics.max_step,
    )
    orbits = []
    for state, period, closure_error, jacobi_drift in zip(
        raw.states, raw.periods, raw.closure_errors, raw.jacobi_drifts, strict=True
    ):
        orbit = Orbit(
            states=np.asarray(state, dtype=float).reshape(1, -1),
            times=np.array([0.0]),
            system=dynamics.system,
        )
        orbit.period = float(period)
        orbit.family_type = family_type
        orbit.closure_error = float(closure_error)
        orbit.parameters["libration_point"] = libration_point
        orbit.parameters["jacobi_drift"] = float(jacobi_drift)
        orbits.append(orbit)

    family = OrbitFamily(orbits=orbits, family_type=family_type, system=dynamics.system)
    family.metadata["planar_pal"] = "rust-full-period-pal"
    return ContinuationResult(
        status=raw.status,
        cause=raw.cause,
        message=raw.message,
        family=family,
        steps=raw.steps,
        step_size=raw.step_size,
    )
