"""统一 Rust 轨道族生成结果的领域适配器。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ...data.templates import ConvergenceState, FailureCause
from ...data.templates.seed import MOON_RADIUS_KM
from ...data.types.orbit import Orbit, OrbitFamily
from ...integrators import generate_cr3bp_family_py, generate_cr3bp_family_windows_py
from ..dynamics import CR3BP_Dynamics
from ..results import FamilyGenerationResult


def generate_rust_family(
    family_type: str,
    libration_point: int,
    n_orbits: int,
    dynamics: CR3BP_Dynamics,
    **parameters: Any,
) -> FamilyGenerationResult:
    """单次调用 Rust 生成轨道族，并重包为领域对象。"""
    characteristic_length = dynamics.system.characteristic_length
    if characteristic_length is None:
        raise ValueError("CR3BP system 尚未设置特征长度")
    raw = generate_cr3bp_family_py(
        family_type=family_type,
        mu=float(dynamics.system.mu),
        characteristic_length_km=float(characteristic_length),
        secondary_radius_km=MOON_RADIUS_KM,
        point=libration_point,
        n_orbits=n_orbits,
        rtol=dynamics.rtol,
        atol=dynamics.atol,
        max_step=dynamics.max_step,
        **parameters,
    )
    return _wrap_outcome(family_type, libration_point, raw, dynamics, parameters)


def generate_rust_family_windows(
    family_type: str,
    libration_point: int,
    n_orbits: int,
    jacobi_windows: Sequence[Sequence[float]],
    dynamics: CR3BP_Dynamics | None = None,
    **parameters: Any,
) -> list[FamilyGenerationResult]:
    """按 Jacobi 能量窗口单次调用 Rust 批量生成轨道族（trace 只走一次）。

    同一组生成参数的延拓 trace 在 Rust 侧只生成一次，各窗口分别筛选
    成员；返回与 ``jacobi_windows`` 同序的结果列表（每窗口一条）。窗口
    零成员时该窗口结果为零成员的结构化软失败（状态可查）。
    """
    if dynamics is None:
        from .cr3bp_orbits import earth_moon_system

        dynamics = CR3BP_Dynamics(earth_moon_system())
    characteristic_length = dynamics.system.characteristic_length
    if characteristic_length is None:
        raise ValueError("CR3BP system 尚未设置特征长度")
    raw_outcomes = generate_cr3bp_family_windows_py(
        family_type=family_type,
        mu=float(dynamics.system.mu),
        characteristic_length_km=float(characteristic_length),
        secondary_radius_km=MOON_RADIUS_KM,
        point=libration_point,
        n_orbits=n_orbits,
        jacobi_windows=[[float(lower), float(upper)] for lower, upper in jacobi_windows],
        rtol=dynamics.rtol,
        atol=dynamics.atol,
        max_step=dynamics.max_step,
        **parameters,
    )
    return [
        _wrap_outcome(family_type, libration_point, raw, dynamics, parameters)
        for raw in raw_outcomes
    ]


def _wrap_outcome(
    family_type: str,
    libration_point: int,
    raw: dict[str, Any],
    dynamics: CR3BP_Dynamics,
    parameters: dict[str, Any],
) -> FamilyGenerationResult:
    """Rust 单条族生成结果 dict → 领域对象。"""
    periodic = raw["periodicity"] == "periodic"
    members = []
    for item in raw["members"]:
        orbit = Orbit(
            states=np.asarray(item["states"], dtype=float),
            times=np.asarray(item["times"], dtype=float),
            system=dynamics.system,
        )
        orbit.period = None if item["period"] is None else float(item["period"])
        orbit.family_type = family_type
        orbit.is_periodic = periodic
        orbit.closure_error = (
            None if item["closure_error"] is None else float(item["closure_error"])
        )
        orbit.periodicity_error = orbit.closure_error
        orbit.parameters = _member_parameters(
            family_type,
            libration_point,
            item,
            parameters,
        )
        members.append(orbit)

    family = OrbitFamily(
        orbits=members,
        family_type=family_type,
        system=dynamics.system,
    )
    family.metadata.update(_family_metadata(family_type, raw["periodicity"], parameters))
    return FamilyGenerationResult(
        status=ConvergenceState(raw["status"]),
        cause=FailureCause(raw["cause"]),
        message=str(raw["message"]),
        family=family,
        requested_members=int(raw["requested_members"]),
        generated_members=int(raw["generated_members"]),
    )


def _member_parameters(
    family_type: str,
    libration_point: int,
    item: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {"libration_point": libration_point}
    if family_type == "halo":
        result.update(
            halo_class=0 if request["max_amplitude_km"] > 0.0 else 1,
            amplitude_z=abs(float(item["states"][0][2])),
        )
    elif family_type == "nrho":
        result.update(
            halo_class=0 if request["north_south"] == 1 else 1,
            perilune_height_km=float(item["perilune_height_km"]),
        )
    elif family_type == "axial":
        result["amplitude_z_km"] = float(item["amplitude_km"])
    elif family_type == "lissajous":
        fraction = float(item["sampling_fraction"])
        result.update(
            amplitude_in_km=fraction * request["amplitude_in_km"],
            amplitude_out_km=fraction * request["amplitude_out_km"],
            phase_in=request["phase_in"],
            phase_out=request["phase_out"],
            sampling_fraction=fraction,
        )
    else:
        result["amplitude_km"] = float(item["amplitude_km"])
        for key in (
            "jacobi_drift",
            "newton_iterations",
            "tangent_system_rank",
            "tangent_system_condition",
            "augmented_system_rank",
            "augmented_system_condition",
            "step_size",
        ):
            if item[key] is not None:
                result[key] = item[key]
    return result


def _family_metadata(
    family_type: str,
    periodicity: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"periodicity": periodicity, "backend": "rust"}
    if family_type == "halo":
        metadata["max_amplitude_km"] = request["max_amplitude_km"]
    elif family_type == "nrho":
        metadata.update(
            perilune_height_max_km=request["perilune_height_max_km"],
            continuation_direction="toward-moon",
        )
    elif family_type == "axial":
        metadata.update(
            max_amplitude_km=request["max_amplitude_km"],
            continuation_direction="increase-amplitude",
        )
    elif family_type == "lissajous":
        metadata["sampling"] = "linear-amplitudes"
    else:
        metadata.update(
            amplitude_range_km=[
                request["min_amplitude_km"],
                request["max_amplitude_km"],
            ],
            continuation_direction=request["continuation_direction"],
            match_tolerance_km=request["match_tolerance_km"],
            planar_pal="rust-full-period-pal",
        )
    return metadata
