"""catalog 数据层测试共享辅助：合成记录工厂。"""

from __future__ import annotations

from typing import Any

import numpy as np


def make_record(
    *,
    orbit_family: str | None = "nrho",
    libration_point: int | None = 2,
    jacobi: tuple[float, float] | None = (3.05, 3.05),
    amplitude: tuple[float, float] | None = (65000.0, 65000.0),
    with_cr3bp: bool = True,
    with_ephemeris: bool = True,
    status: str = "converged",
    cause: str = "none",
    message: str = "任务完成",
    source_tool: str = "design_orbit",
    source_record_id: str | None = None,
    family_id: str | None = None,
    member_index: int | None = None,
    tags: list[str] | None = None,
    note: str = "",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """构造一条合成轨道记录（meta + arrays），供存储引擎测试。

    默认形态为设计产物：CR3BP 段与星历段双段并存（一轨一记录，
    ADR 0045）。``with_cr3bp=False`` 得到站保产物形态（仅星历段）；
    ``family_id``/``member_index`` 给出时为族成员记录。
    """
    arrays: dict[str, np.ndarray] = {}
    if with_cr3bp:
        arrays["cr3bp/states"] = np.tile(np.arange(6, dtype=float), (10, 1))
        arrays["cr3bp/times"] = np.linspace(0.0, 1.0, 10)
    if with_ephemeris:
        n = 5
        arrays.update(
            {
                "eph/year": np.full(n, 2024),
                "eph/month": np.full(n, 1),
                "eph/day": np.arange(1, n + 1),
                "eph/hour": np.zeros(n, dtype=int),
                "eph/minute": np.zeros(n, dtype=int),
                "eph/second": np.zeros(n),
                "eph/position_km": np.arange(n * 3, dtype=float).reshape(n, 3),
                "eph/velocity_mps": np.full((n, 3), 1000.0),
                "eph/synodic_position": np.full((n, 3), 0.5),
            }
        )
    meta: dict[str, Any] = {
        "source_tool": source_tool,
        "source_record_id": source_record_id,
        "family_id": family_id,
        "member_index": member_index,
        "classification": {
            "orbit_family": orbit_family,
            "libration_point": libration_point,
            "jacobi": list(jacobi) if jacobi is not None else None,
            "amplitude": list(amplitude) if amplitude is not None else None,
            "has_cr3bp": with_cr3bp,
            "has_ephemeris": with_ephemeris,
        },
        "status": status,
        "cause": cause,
        "message": message,
        "scalars": {},
        "request": {"orbit_type": "NRHO"},
        "tags": list(tags) if tags else [],
        "note": note,
    }
    return meta, arrays
