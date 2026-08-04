"""轨迹数据容器：EphemerisTable（通用星历容器）与 NominalOrbit（名义轨道契约）。

- ``EphemerisTable``：UTC + GCRS 位置 km/速度 m/s + 会合系位置，通用容器，非 DFH
  专属（ADR 0011 迁移，源：``io/ephemeris.py``）。DFH 格式读写（parse/write）
  保留在 ``io/`` 作临时脚本。
- ``NominalOrbit``：FR1↔FR2 数据契约（ADR 0015，Gómez vol I §8.2.3）：等间距历元
  状态表 + Floquet 基 + 投影因子表 + 高次插值器。Floquet 基 + 投影因子由 FR1
  预计算，控制全程插值不复算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["EphemerisTable", "NominalOrbit"]


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
