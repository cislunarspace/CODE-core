"""轨道预报：给定初值与力模型的高精度数值外推。

单段能力（不建独立编排器，ADR 0011）：配 ForceModel + 调 propagate + 输出
EphemerisTable。单文件模块（不是目录）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.types import EphemerisTable

__all__ = ["propagate_orbit"]

#: J2000 历元的 TDB 儒略日（SPICE ET 定义为相对此历元的 TDB 秒）
_J2000_JD_TDB = 2451545.0
_SECONDS_PER_DAY = 86400.0

#: 默认三体力模型（地月日点质量引力）
_DEFAULT_FORCE_CONFIG: dict[str, Any] = {
    "version": 1,
    "forces": [
        {
            "name": "gravity_earth",
            "type": "PointMassGravity",
            "enabled": True,
            "params": {"body": "EARTH", "mu": 398600.435507},
        },
        {
            "name": "gravity_moon",
            "type": "PointMassGravity",
            "enabled": True,
            "params": {"body": "MOON", "mu": 4902.800118},
        },
        {
            "name": "gravity_sun",
            "type": "PointMassGravity",
            "enabled": True,
            "params": {"body": "SUN", "mu": 1.32712440018e11},
        },
    ],
}


def propagate_orbit(
    initial_state: Any,
    epoch: Any,
    duration: float,
    force_config: dict[str, Any] | None = None,
    output_step: float = 3600.0,
    **kwargs,
) -> EphemerisTable:
    """高精度轨道预报。

    配 ForceModel 并传播，输出通用星历表容器（UTC + GCRS 位置/速度）。

    Args:
        initial_state: 初值（GCRS，km, km/s，形状 (6,)）。
        epoch: 起始历元 UTC（ISO 字符串或 ``[年,月,日,时,分,秒]``）。
        duration: 预报时长（秒）。
        force_config: 力模型配置（缺省用默认三体力模型）。
        output_step: 输出间隔（秒）。
        kwargs: 传给 ForceModel 的额外配置（如 system/spice 等）。

    Returns:
        预报星历表。

    Raises:
        ValueError: 初值形状或时长非法。
    """
    state = np.asarray(initial_state, dtype=float)
    if state.shape != (6,):
        raise ValueError(f"initial_state 应为 (6,)，实际 {state.shape}")
    if float(duration) <= 0:
        raise ValueError(f"duration 必须为正数，当前 {duration}")

    from ..data.kernels.manager import SPICEManager
    from .coordinate.coordinate_system import CoordinateSystem
    from .coordinate.standard_axes import ICRSAxes
    from .coordinate.standard_origins import CelestialBodyOrigin
    from .dynamics.ephemeris_system import EphemerisSystem
    from .forces import ForceModel

    spice = SPICEManager()
    from .design.design_orbit import load_design_kernels

    load_design_kernels(spice, kwargs.get("kernel_dir"))

    if force_config is None:
        force_config = _DEFAULT_FORCE_CONFIG
    bodies = _extract_bodies(force_config)
    if not bodies:
        bodies = ["EARTH", "MOON", "SUN"]

    system = EphemerisSystem(bodies=bodies, spice=spice, origin="EARTH")
    system.coordinate_system = CoordinateSystem(
        axes=ICRSAxes(),
        origin=CelestialBodyOrigin(body="EARTH", spice=spice),
    )
    fm = ForceModel.from_config(force_config, system)

    if isinstance(epoch, str):
        et0 = spice.utc_to_et(epoch)
    else:
        epoch_parts = list(epoch)
        if len(epoch_parts) == 6:
            y, mo, d, h, mi, s = epoch_parts
            epoch_iso = (
                f"{int(y):04d}-{int(mo):02d}-{int(d):02d}T{int(h):02d}:{int(mi):02d}:{s:06.3f}"
            )
            et0 = spice.utc_to_et(epoch_iso)
        else:
            raise ValueError(f"不支持的 epoch 格式: {epoch}")

    etf = et0 + float(duration)
    t_eval = np.arange(et0, etf + 0.5 * float(output_step), float(output_step))

    result = fm.propagate(state, t_span=(et0, etf), t_eval=t_eval)

    times = result["time"]
    states = result["states"]
    n = len(times)

    years = np.empty(n, dtype=int)
    months = np.empty(n, dtype=int)
    days = np.empty(n, dtype=int)
    hours = np.empty(n, dtype=int)
    minutes = np.empty(n, dtype=int)
    seconds = np.empty(n, dtype=float)
    for i, et in enumerate(times):
        utc_str = spice.et_to_utc(float(et))
        y, mo, d, h, mi, s = _parse_utc_calendar(utc_str)
        years[i] = y
        months[i] = mo
        days[i] = d
        hours[i] = h
        minutes[i] = mi
        seconds[i] = s

    return EphemerisTable(
        year=years,
        month=months,
        day=days,
        hour=hours,
        minute=minutes,
        second=seconds,
        position_km=states[:, :3].copy(),
        velocity_mps=states[:, 3:6].copy() * 1000.0,
        synodic_position=np.zeros((n, 3)),
        times_jd_tdb=(_J2000_JD_TDB + times / _SECONDS_PER_DAY).copy(),
    )


def _extract_bodies(force_config: dict[str, Any]) -> list[str]:
    """从 force_config 提取唯一天体列表。"""
    bodies: list[str] = []
    seen: set[str] = set()
    for entry in force_config.get("forces", []):
        body = entry.get("params", {}).get("body")
        if body:
            name = str(body).upper()
            if name not in seen:
                seen.add(name)
                bodies.append(name)
    return bodies


def _parse_utc_calendar(utc_str: str) -> tuple[int, int, int, int, int, float]:
    """把 SPICE et2utc ISO-C 输出拆为 (y, mo, d, h, mi, s)。"""
    date_part, time_part = utc_str.split("T")
    y, mo, d = date_part.split("-")
    h, mi, s = time_part.split(":")
    return int(y), int(mo), int(d), int(h), int(mi), float(s)
