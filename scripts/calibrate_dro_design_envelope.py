"""DRO 轨道设计振幅包络标定与初值核验（#840）。

用法：
    .venv/bin/python scripts/calibrate_dro_design_envelope.py
    .venv/bin/python scripts/calibrate_dro_design_envelope.py \
        --amplitudes 175000 200000
    .venv/bin/python scripts/calibrate_dro_design_envelope.py --verify-only

默认按 5000 km 步长标定 150000 到 200000 km 的 11 个采样点，与 ADR 0061
修订记录的网格一致。

每个振幅首次运行 ``design_orbit``，把 CR3BP 初值、修正后首状态与一年星历指标
写进 ``scripts/.cache/dro_design_envelope.json``。重复运行时读取记录，用记录
的初值重新传播并核对指标，不再做多重打靶。

CR3BP 轨道由振幅唯一确定，记录值按一周期闭合核验。星历初值依赖历元与力模型，
记录里保存完整配置，核验时按同一配置重建力模型。星历修正的收敛残差只约束
修正弧段，不约束长期外推。本脚本同时报告一年月距有界性，并按设计振幅的
三倍判为逃逸。

不要在 scripts/ 里用 uv run，它会触发 editable 重建。
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from e2m2e.algorithm.coordinate.synodic_j2000 import SynodicJ2000System
from e2m2e.algorithm.design.design_orbit import (
    DEFAULT_DESIGN_PERTURBATION,
    _build_design_force_model,
    _build_ephemeris_table,
    design_orbit,
    earth_moon_system,
    load_design_kernels,
)
from e2m2e.algorithm.design.tuning import DRO_EPHEMERIS_ESCAPE_FACTOR
from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.api.models import DesignOrbitRequest
from e2m2e.data.kernels.manager import SPICEManager
from e2m2e.data.templates import CHAR_LENGTH_KM, CHAR_PERIOD_SEC

_RECORD_PATH = Path(__file__).resolve().parent / ".cache" / "dro_design_envelope.json"
_DEFAULT_AMPLITUDES = tuple(150000.0 + 5000.0 * step for step in range(11))


def _request(amplitude_km: float) -> DesignOrbitRequest:
    request = DesignOrbitRequest(
        orbit_type="DRO",
        amplitude=150000.0,
        epoch=(2024, 1, 1, 0, 0, 0.0),
    )
    # 200000 km 以上的标定点超出公开包络，只在本诊断脚本里绕过条件校验
    object.__setattr__(request, "amplitude", float(amplitude_km))
    return request


def _moon_dist_km(ephemeris: Any) -> np.ndarray:
    """会合系为地心归一，月球在 x=1。"""
    synodic = np.asarray(ephemeris.synodic_position, dtype=float)
    return np.linalg.norm(synodic - np.array([1.0, 0.0, 0.0]), axis=1) * CHAR_LENGTH_KM


def _year_metrics(ephemeris: Any, period: float, output_step: float) -> dict[str, Any]:
    dist = _moon_dist_km(ephemeris)
    period_s = float(period) * CHAR_PERIOD_SEC / (2.0 * np.pi)
    per_rev = max(1, int(round(period_s / float(output_step))))
    rev_amplitudes = []
    for k in range(len(dist) // per_rev):
        segment = dist[k * per_rev : (k + 1) * per_rev]
        if len(segment) > 1:
            rev_amplitudes.append(float(0.5 * (segment.min() + segment.max())))
    return {
        "moon_dist_min_km": float(dist.min()),
        "moon_dist_max_km": float(dist.max()),
        "amplitude_km": float(0.5 * (dist.min() + dist.max())),
        "rev_amplitudes_km": rev_amplitudes,
    }


def _design(amplitude_km: float) -> tuple[dict[str, Any], float]:
    request = _request(amplitude_km)
    start = time.perf_counter()
    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = design_orbit(request)
    except Exception as exc:  # noqa: BLE001 - 诊断脚本记录任意失败
        elapsed = time.perf_counter() - start
        return {
            "amplitude_km": float(amplitude_km),
            "error": f"{type(exc).__name__}: {exc}",
            "warnings": [str(item.message) for item in caught],
            "elapsed_s": elapsed,
        }, elapsed
    elapsed = time.perf_counter() - start

    orbit = result.cr3bp_orbit
    assert orbit is not None and orbit.period is not None
    assert result.correction is not None
    perturbation = dict(
        DEFAULT_DESIGN_PERTURBATION if request.perturbation is None else request.perturbation
    )
    record = {
        "amplitude_km": float(amplitude_km),
        "epoch_utc": result.epoch_utc,
        "duration": float(request.duration),
        "output_step": float(request.output_step),
        "perturbation": perturbation,
        "earth_degree": int(request.earth_degree),
        "moon_degree": int(request.moon_degree),
        "correction_method": result.correction_method,
        "correction_revolutions": int(request.correction_revolutions),
        "result_status": getattr(result.status, "name", str(result.status)),
        "cause": getattr(result.cause, "name", str(result.cause)),
        "message": result.message,
        "cr3bp": {
            "x0": float(orbit.states[0, 0]),
            "vy0": float(orbit.states[0, 4]),
            "period": float(orbit.period),
        },
        "t0_corr": float(result.correction.t_patch[0]),
        "initial_state": [float(x) for x in result.initial_state],
        "correction": {
            "status": getattr(result.correction.status, "name", str(result.correction.status)),
            "iterations": int(result.correction.iterations),
            "position_residual_km": float(result.correction.max_residual),
            "velocity_residual_km_s": float(result.correction.velocity_residual),
        },
        "year": _year_metrics(result.ephemeris, orbit.period, request.output_step),
        "warnings": [str(item.message) for item in caught],
        "elapsed_s": elapsed,
    }
    record["bounded"] = bool(
        record["year"]["moon_dist_max_km"] <= DRO_EPHEMERIS_ESCAPE_FACTOR * float(amplitude_km)
    )
    return record, elapsed


def _verify(record: dict[str, Any]) -> tuple[bool, str]:
    if record.get("error") or "cr3bp" not in record:
        return False, f"记录为失败设计：{record.get('error', '缺 cr3bp 字段')}"
    system = earth_moon_system()
    dynamics = CR3BP_Dynamics(system)
    cr3bp = record["cr3bp"]
    state = np.array([cr3bp["x0"], 0.0, 0.0, 0.0, cr3bp["vy0"], 0.0], dtype=float)
    closure = dynamics.propagate(state, (0.0, cr3bp["period"]), t_eval=[cr3bp["period"]])
    final = np.asarray(closure["states"], dtype=float)[-1]
    error = float(np.linalg.norm(final - state))
    if error > 1e-8:
        return False, f"CR3BP 一周期闭合误差 {error:.3e} 超过 1e-8"

    spice = SPICEManager()
    load_design_kernels(spice)
    fm, _, _, bodies = _build_design_force_model(
        spice,
        record["perturbation"],
        earth_degree=record["earth_degree"],
        moon_degree=record["moon_degree"],
        dyb=None,
    )
    et0 = spice.utc_to_et(record["epoch_utc"])
    et_grid = et0 + np.arange(
        0.0, record["duration"] + 0.5 * record["output_step"], record["output_step"]
    )
    t0_corr = float(record["t0_corr"])
    margin = float(cr3bp["period"]) * float(system.characteristic_time)
    perturber_map = {"EARTH": ["SUN", "MOON"], "MOON": ["EARTH"]}
    perturbers = {p for body in bodies for p in perturber_map.get(body.upper(), [])}
    cache_bodies = list(dict.fromkeys([*bodies, *perturbers]))
    spice.enable_ephem_cache(
        cache_bodies,
        float(min(et0, t0_corr)) - margin,
        float(max(et_grid[-1], t0_corr)) + margin,
        dt=3600.0,
        observer="EARTH",
        frame_pairs=[("ITRF93", "J2000"), ("MOON_PA", "J2000")],
    )
    try:
        out = fm.propagate(
            np.asarray(record["initial_state"], dtype=float),
            (min(t0_corr, et0), float(et_grid[-1])),
            t_eval=et_grid,
            max_steps=2_000_000,
        )
    finally:
        spice.disable_ephem_cache()

    syn_j2000 = SynodicJ2000System(cr3bp_system=system, spice=spice)
    ephemeris = _build_ephemeris_table(
        spice, syn_j2000, et0, et_grid, np.asarray(out["states"], dtype=float)
    )
    metrics = _year_metrics(ephemeris, cr3bp["period"], record["output_step"])
    old_max = float(record["year"]["moon_dist_max_km"])
    new_max = float(metrics["moon_dist_max_km"])
    bounded = bool(
        record.get(
            "bounded",
            old_max <= DRO_EPHEMERIS_ESCAPE_FACTOR * float(record["amplitude_km"]),
        )
    )
    expected_status = "CONVERGED" if bounded else "DIVERGED"
    recorded_status = record.get("result_status")
    if recorded_status is not None and recorded_status != expected_status:
        return False, f"记录状态 {recorded_status} 与一年星历有界性 {expected_status} 不符"
    if bounded:
        if abs(new_max - old_max) > 1e-6 * max(1.0, abs(old_max)):
            return False, f"一年星历最大月距 {new_max:.1f} km 与记录 {old_max:.1f} km 不符"
        return True, f"核验通过，最大月距 {new_max:.1f} km"
    # 发散轨道整年指标对初值敏感，无法逐位复现。核验前 3 圈幅值，
    # 确认记录的首状态仍在同一条局部轨道上，并如实报告整年不可复现。
    old_rev = record["year"]["rev_amplitudes_km"][:3]
    new_rev = metrics["rev_amplitudes_km"][:3]
    if len(old_rev) != len(new_rev) or not old_rev:
        return False, "记录缺少前 3 圈幅值，无法核验"
    for old, new in zip(old_rev, new_rev, strict=True):
        if abs(new - old) > 1e-3 * max(1.0, abs(old)):
            return False, f"前 3 圈幅值 {new:.1f} km 与记录 {old:.1f} km 不符"
    return True, "前 3 圈核验通过，整年星历发散且不可复现"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, records: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amplitudes",
        nargs="+",
        type=float,
        default=list(_DEFAULT_AMPLITUDES),
        help="标定振幅，km，默认 150000 与 200000",
    )
    parser.add_argument("--records", type=Path, default=_RECORD_PATH)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只核验已有记录，不做新设计",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有记录，重新设计并覆盖",
    )
    args = parser.parse_args()

    records = _load(args.records)
    failures = 0
    print(
        f"{'振幅':>9} {'模式':>6} {'状态':>10} {'迭代':>4} {'残差 km':>10} "
        f"{'月距最大':>10} {'有界':>4} {'耗时 s':>7}"
    )
    for amplitude in args.amplitudes:
        key = f"{amplitude:.3f}"
        record = records.get(key)
        if record is not None and not args.force:
            ok, message = _verify(record)
            mode = "核验"
            if not ok:
                failures += 1
                print(f"{amplitude:9.0f} {mode:>6} 失败 {message}")
                continue
            year = record["year"]
            bounded = record.get(
                "bounded",
                year["moon_dist_max_km"] <= DRO_EPHEMERIS_ESCAPE_FACTOR * amplitude,
            )
            if not bounded:
                failures += 1
            status = record.get("result_status", record["correction"]["status"])
            print(
                f"{amplitude:9.0f} {mode:>6} {status:>10} "
                f"{record['correction']['iterations']:4d} "
                f"{record['correction']['position_residual_km']:10.3e} "
                f"{year['moon_dist_max_km']:10.1f} {str(bounded):>4} "
                f"{record['elapsed_s']:7.1f}  {message}"
            )
            continue
        if args.verify_only:
            failures += 1
            print(f"{amplitude:9.0f} {'核验':>6} 缺记录")
            continue
        record, elapsed = _design(amplitude)
        records[key] = record
        _save(args.records, records)
        if record.get("error"):
            failures += 1
            print(f"{amplitude:9.0f} {'设计':>6} 失败 {record['error']}")
            continue
        if not record["bounded"]:
            failures += 1
        warn = " 告警" if record["warnings"] else ""
        status = record.get("result_status", record["correction"]["status"])
        print(
            f"{amplitude:9.0f} {'设计':>6} {status:>10} "
            f"{record['correction']['iterations']:4d} "
            f"{record['correction']['position_residual_km']:10.3e} "
            f"{record['year']['moon_dist_max_km']:10.1f} "
            f"{str(record['bounded']):>4} {elapsed:7.1f}{warn}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
