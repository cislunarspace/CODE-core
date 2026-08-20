"""产物型 Facade 结果 → catalog 记录的构建纯函数（ADR 0031 决策 2/3/8）。

分类字段在生成时填写（不做事后推断）：构建器同时拿着原始请求模型与
算法层结果，族类型、平动点、能量都是已知量。主振幅统一为几何主振幅
（CR3BP 段位置三分量半极差最大值 × 特征长度，km）；各族定义不一的
参数振幅保留在 ``request`` 快照中，不进分类字段。

无产物时不建记录（返回 ``None``）：族零成员、站保全样本失败（受控
星历缺失）都不产生记录。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from e2m2e.data.catalog import (
    cr3bp_segment_arrays,
    ephemeris_segment_arrays,
    geometric_amplitude_km,
    member_array_key,
    point_interval,
)
from e2m2e.data.templates import ConvergenceState, FailureCause

if TYPE_CHECKING:
    from e2m2e.data.types.trajectory import EphemerisTable

__all__ = ["build_design_record", "build_family_record", "build_control_record"]

#: design_orbit 的 orbit_type → (orbit_family, libration_point)。
#: HALO/NRHO/LISSAJOUS/AXIAL 的平动点取请求的 collinear_point，不在此表。
_DESIGN_FAMILY_POINT: dict[str, tuple[str, int | None]] = {
    "DRO": ("dro", None),
    "DPO": ("dpo", None),
    "L4": ("lissajous", 4),
    "L5": ("lissajous", 5),
    "L4_SPO": ("spo", 4),
    "L5_SPO": ("spo", 5),
    "L4_LPO": ("lpo", 4),
    "L5_LPO": ("lpo", 5),
    "L4_HORSESHOE": ("horseshoe", 4),
    "L5_HORSESHOE": ("horseshoe", 5),
    "ELFO": ("elfo", None),
}


def build_design_record(request: Any, result: Any) -> tuple[dict, dict[str, np.ndarray]] | None:
    """从 design_orbit 的请求与结果构建记录（双段并存）。

    Args:
        request: ``DesignOrbitRequest``（已校验、含默认值）。
        result: ``OrbitDesignResult``。
    """
    arrays: dict[str, np.ndarray] = {}
    cr3bp_orbit = result.cr3bp_orbit
    char_length_km: float | None = None
    mu: float | None = None
    amplitude: list[float] | None = None
    if cr3bp_orbit is not None:
        arrays.update(cr3bp_segment_arrays(cr3bp_orbit.states, cr3bp_orbit.times))
        system = cr3bp_orbit.system
        char_length_km = getattr(system, "characteristic_length", None)
        mu = getattr(system, "mu", None)
        amplitude = point_interval(geometric_amplitude_km(cr3bp_orbit.states, char_length_km))
    if result.ephemeris is not None:
        arrays.update(ephemeris_segment_arrays(result.ephemeris))
    if not arrays:
        return None

    selection = request.orbit_type.upper()
    if selection in _DESIGN_FAMILY_POINT:
        orbit_family, libration_point = _DESIGN_FAMILY_POINT[selection]
    else:
        orbit_family, libration_point = selection.lower(), request.collinear_point

    jacobi = _finite_or_none(result.cr3bp_jacobi)
    correction = result.correction
    meta = _base_meta(
        source_tool="design_orbit",
        source_record_id=None,
        classification={
            "orbit_family": orbit_family,
            "libration_point": libration_point,
            "jacobi": point_interval(jacobi),
            "amplitude": amplitude,
            "has_cr3bp": cr3bp_orbit is not None,
            "has_ephemeris": result.ephemeris is not None,
        },
        status=result.status,
        cause=result.cause,
        message=result.message,
        scalars={
            "member_count": 1 if cr3bp_orbit is not None else 0,
            "orbit_type": selection,
            "epoch_utc": result.epoch_utc,
            "duration_day": result.duration_day,
            "output_step_sec": result.output_step_sec,
            "mu": mu,
            "char_length_km": char_length_km,
            "iterations": correction.iterations if correction is not None else 0,
            "correction_method": result.correction_method,
        },
        request=_request_snapshot(request),
    )
    return meta, arrays


def build_family_record(
    request: Any,
    *,
    family: Any,
    status: ConvergenceState,
    cause: FailureCause,
    message: str,
    requested_members: int,
    generated_members: int,
) -> tuple[dict, dict[str, np.ndarray]] | None:
    """从 orbit_family_generation 的请求与族结果构建记录（一族一条）。

    成员参数（周期、闭合误差、Jacobi、参数表）在元数据 ``members`` 内，
    成员数组（states/times）在 ``cr3bp/members/`` 段。零成员时不建记录。

    Args:
        request: ``FamilyGenerationRequest``（已校验、含默认值）。
        family: ``OrbitFamily``（或读取接口兼容的响应对象）。
    """
    members = list(family.orbits)
    if not members:
        return None
    system = family.system
    char_length_km = getattr(system, "characteristic_length", None)
    mu = getattr(system, "mu", None)

    arrays: dict[str, np.ndarray] = {}
    member_metas: list[dict[str, Any]] = []
    jacobis: list[float] = []
    amplitudes: list[float] = []
    for index, orbit in enumerate(members):
        arrays[member_array_key(index, "states")] = np.asarray(orbit.states, dtype=float)
        arrays[member_array_key(index, "times")] = np.asarray(orbit.times, dtype=float)
        jacobi = _member_jacobi(system, orbit)
        amplitude_km = geometric_amplitude_km(orbit.states, char_length_km)
        if jacobi is not None:
            jacobis.append(jacobi)
        if amplitude_km is not None:
            amplitudes.append(amplitude_km)
        member_metas.append(
            {
                "index": index,
                "period": _finite_or_none(orbit.period),
                "closure_error": _finite_or_none(getattr(orbit, "closure_error", None)),
                "jacobi": jacobi,
                "amplitude_km": amplitude_km,
                "amplitudes": _sanitize_value(getattr(orbit, "amplitudes", {})),
                "parameters": _sanitize_value(getattr(orbit, "parameters", {})),
            }
        )

    meta = _base_meta(
        source_tool="orbit_family_generation",
        source_record_id=None,
        classification={
            "orbit_family": family.family_type,
            "libration_point": request.libration_point,
            "jacobi": _envelope(jacobis),
            "amplitude": _envelope(amplitudes),
            "has_cr3bp": True,
            "has_ephemeris": False,
        },
        status=status,
        cause=cause,
        message=message,
        scalars={
            "member_count": len(members),
            "requested_members": requested_members,
            "generated_members": generated_members,
            "periodicity": family.metadata.get("periodicity", "periodic"),
            "mu": mu,
            "char_length_km": char_length_km,
        },
        request=_request_snapshot(request),
    )
    meta["members"] = member_metas
    return meta, arrays


def build_control_record(
    request: Any,
    result: Any,
    *,
    source_meta: dict[str, Any] | None = None,
) -> tuple[dict, dict[str, np.ndarray]] | None:
    """从 control_orbit 的请求与结果构建记录（只含星历段）。

    谱系：``source_meta`` 给出被控轨道记录的元数据时（经
    ``input_record_id`` 输入），``source_record_id`` 指向它并继承其
    族/平动点分类；裸星历输入时谱系为空。全样本失败（受控星历缺失）
    不建记录。

    Args:
        request: ``ControlOrbitRequest``。
        result: ``ControlOrbitResult``。
        source_meta: 被控轨道记录的元数据（可选）。
    """
    ephemeris: EphemerisTable | None = result.controlled_ephemeris
    if ephemeris is None:
        return None
    arrays: dict[str, np.ndarray] = ephemeris_segment_arrays(ephemeris)
    arrays["result/maneuvers_mjd_tdb"] = np.asarray(result.maneuvers.mjd_tdb, dtype=float)
    arrays["result/maneuvers_delta_v_mps"] = np.asarray(result.maneuvers.delta_v_mps, dtype=float)
    arrays["result/sk_rows"] = np.asarray(result.sk_statistic.rows, dtype=float)

    source_classification = (source_meta or {}).get("classification", {})
    meta = _base_meta(
        source_tool="control_orbit",
        source_record_id=request.input_record_id,
        classification={
            "orbit_family": source_classification.get("orbit_family"),
            "libration_point": source_classification.get("libration_point"),
            "jacobi": None,
            "amplitude": None,
            "has_cr3bp": False,
            "has_ephemeris": True,
        },
        status=result.status,
        cause=result.cause,
        message=result.message,
        scalars={
            "member_count": 0,
            "control_mode": request.control_mode,
            "num_failed": result.num_failed,
            "mu": request.mu,
            "delta_v_total_mps": float(np.sum(result.maneuvers.delta_v_mps)),
        },
        request=_request_snapshot(request),
    )
    return meta, arrays


def _base_meta(
    *,
    source_tool: str,
    source_record_id: str | None,
    classification: dict[str, Any],
    status: ConvergenceState,
    cause: FailureCause,
    message: str,
    scalars: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    """记录元数据的公共骨架；schema_version/record_id/created_at/arrays
    指针由存储引擎在写入时填写。"""
    return {
        "source_tool": source_tool,
        "source_record_id": source_record_id,
        "classification": classification,
        "status": ConvergenceState(status).value,
        "cause": FailureCause(cause).value,
        "message": message,
        "scalars": scalars,
        "request": request,
        "members": [],
        "tags": [],
        "note": "",
    }


def _request_snapshot(request: Any) -> dict[str, Any]:
    """原始请求模型的 JSON 快照（可追溯、可复算）。

    非 JSON 原生类型的字段值（EphemerisTable、EngineLayout 等）以类型
    标记 + 摘要表示，保证快照可序列化。
    """
    return {key: _sanitize_value(value) for key, value in request.model_dump().items()}


def _sanitize_value(value: Any) -> Any:
    """递归序列化为 JSON 兼容类型；非原生对象给类型标记摘要。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if hasattr(value, "position_km") and hasattr(value, "velocity_mps"):
        return {"_type": "EphemerisTable", "points": len(value.year)}
    return {"_type": type(value).__name__, "repr": repr(value)}


def _member_jacobi(system: Any, orbit: Any) -> float | None:
    """成员初态 Jacobi 常数；系统不具备该能力时为 None。"""
    if system is None or not hasattr(system, "get_jacobi_constant"):
        return None
    return _finite_or_none(system.get_jacobi_constant(orbit.states[0]))


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _envelope(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [min(values), max(values)]
