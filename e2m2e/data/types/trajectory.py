"""轨迹数据容器：EphemerisTable（通用星历容器）与 NominalOrbit（名义轨道契约）。

- ``EphemerisTable``：UTC + GCRS 位置 km/速度 m/s + 会合系位置，通用容器，非 DFH
  专属（ADR 0011 迁移，源：``io/ephemeris.py``）。DFH 文本格式 parse/read/write
  函数与本容器同生命周期，不作独立脚本。
- ``NominalOrbit``：FR1↔FR2 数据契约（ADR 0015，Gómez vol I §8.2.3）：等间距历元
  状态表 + Floquet 基 + 投影因子表 + 高次插值器。Floquet 基 + 投影因子由 FR1
  预计算，控制全程插值不复算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "EphemerisTable",
    "NominalOrbit",
    "parse_ephemeris",
    "read_ephemeris",
    "write_ephemeris",
]


@dataclass
class EphemerisTable:
    """通用星历表容器。

    Attributes:
        year, month, day, hour, minute: 整型数组，形状 ``(n,)``
        second: 秒（含小数），形状 ``(n,)``
        position_km: GCRS 位置（km），形状 ``(n, 3)``
        velocity_mps: GCRS 速度（m/s），形状 ``(n, 3)``
        synodic_position: 地月会合系位置（无量纲），形状 ``(n, 3)``
        raw_text: 原始文件文本；程序生成（非读入）时为空串。
        times_jd_tdb: 历元 TDB 儒略日序列（形状 ``(n,)``），由预报/设计链路
            填充；读入的 DFH 星历可能为 ``None``。JD_TDB = 2451545.0 + ET/86400。
    """

    year: np.ndarray
    month: np.ndarray
    day: np.ndarray
    hour: np.ndarray
    minute: np.ndarray
    second: np.ndarray
    position_km: np.ndarray
    velocity_mps: np.ndarray
    synodic_position: np.ndarray
    raw_text: str = field(default="", repr=False)
    times_jd_tdb: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.year)


@dataclass
class NominalOrbit:
    """名义轨道：FR1（设计）→ FR2（保持）数据契约。

    Attributes:
        epochs: 等间距历元（UTC）。
        states: 等间距历元状态表（GCRS，km, km/s，形状 (n, 6)）。
        synodic_positions: 会合系位置（可选，绘图/特征点用）。
        floquet_basis: Floquet 基向量表（可选，特征点控制用）。
        projection_factors: 投影因子表（可选，开/关控制用）。
        interpolator: 高次插值器（Lagrange r=5~6）。
    """

    epochs: np.ndarray
    states: np.ndarray
    synodic_positions: np.ndarray | None = None
    floquet_basis: np.ndarray | None = None
    projection_factors: np.ndarray | None = None
    interpolator: Any = field(default=None, repr=False)

    def state_at(self, t: float) -> np.ndarray:
        """任意时刻（秒）的标称状态（6 维），经插值器。"""
        if self.interpolator is None:
            raise NotImplementedError("NominalOrbit 插值器待实现")
        raise NotImplementedError  # pragma: no cover - 插值器契约待 FR1 落地


# ===========================================================================
# DFH EPHEMERIDES_*.TXT 文本格式序列化
# ===========================================================================


def parse_ephemeris(raw: str) -> EphemerisTable:
    """解析 EPHEMERIDES_*.TXT 文本，返回 :class:`EphemerisTable`。"""
    epoch_rows: list[list[float]] = []
    pos_rows: list[list[float]] = []
    vel_rows: list[list[float]] = []
    sync_rows: list[list[float]] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        ts = parts[0].split("-")
        if len(ts) < 6:
            continue
        try:
            epoch = [float(t) for t in ts[:6]]
            vals = [float(v) for v in parts[1:10]]
        except ValueError:
            continue
        epoch_rows.append(epoch)
        pos_rows.append(vals[0:3])
        vel_rows.append(vals[3:6])
        sync_rows.append(vals[6:9])

    epochs = np.array(epoch_rows, dtype=float).reshape(-1, 6)
    ints = epochs[:, :5].astype(int) if len(epochs) else np.empty((0, 5), dtype=int)
    return EphemerisTable(
        year=ints[:, 0] if len(epochs) else np.array([], dtype=int),
        month=ints[:, 1] if len(epochs) else np.array([], dtype=int),
        day=ints[:, 2] if len(epochs) else np.array([], dtype=int),
        hour=ints[:, 3] if len(epochs) else np.array([], dtype=int),
        minute=ints[:, 4] if len(epochs) else np.array([], dtype=int),
        second=epochs[:, 5] if len(epochs) else np.array([], dtype=float),
        position_km=np.array(pos_rows, dtype=float).reshape(-1, 3),
        velocity_mps=np.array(vel_rows, dtype=float).reshape(-1, 3),
        synodic_position=np.array(sync_rows, dtype=float).reshape(-1, 3),
        raw_text=raw,
    )


def read_ephemeris(path: str | Path) -> EphemerisTable:
    """从文件读入 EPHEMERIDES_*.TXT。"""
    raw = Path(path).read_text(encoding="utf-8")
    return parse_ephemeris(raw)


def write_ephemeris(table: EphemerisTable, path: str | Path) -> Path:
    """按 DFH 格式写出星历文件（CRLF 行尾，UTF-8 无 BOM），返回写入的文件路径。"""
    lines = []
    for k in range(len(table)):
        head = (
            f"{table.year[k]:04d}-{table.month[k]:02d}-{table.day[k]:02d}"
            f"-{table.hour[k]:02d}-{table.minute[k]:02d}-{table.second[k]:05.2f}"
        )
        nums = (
            list(table.position_km[k])
            + list(table.velocity_mps[k])
            + list(table.synodic_position[k])
        )
        lines.append(head + "".join(f"{v:25.12f}" for v in nums))
    out = Path(path)
    out.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return out
