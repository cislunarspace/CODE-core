"""地球定向参数（EOP）文件解析（GMAT C04 格式）。

数据层（ADR 0011 迁移，源：``core/coordinate/gmat_eop.py`` 的 EOP 部分 +
``core/coordinate/gmat_data.py`` 的 ``CoordinateDataError``）。只留**数据
解析**；EOP → ITRF 的转换算法在 ``algorithm/coordinate/``。

GMAT C04 EOP reader：只消费 ``mjd, x, y, UT1-UTC, LOD``；后续
``dPsi/dEps`` 字段保留在原文件中但不进入当前 native ITRF 链路。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...exceptions import E2M2EError

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
JD_MJD_OFFSET = 2400000.5


class CoordinateDataError(E2M2EError, RuntimeError):
    """坐标数据缺失、越界或格式错误。"""


@dataclass(frozen=True)
class EopRecord:
    year: int
    month: int
    day: int
    mjd: float
    x_arcsec: float
    y_arcsec: float
    ut1_utc: float
    lod: float

    @property
    def x_rad(self) -> float:
        return self.x_arcsec * ARCSEC_TO_RAD

    @property
    def y_rad(self) -> float:
        return self.y_arcsec * ARCSEC_TO_RAD


@dataclass(frozen=True)
class EopSample:
    mjd: float
    x_rad: float
    y_rad: float
    ut1_utc: float
    lod: float


class EopFile:
    """EOP 文件解析与按 UTC MJD 查询。"""

    def __init__(self, records: tuple[EopRecord, ...]) -> None:
        if not records:
            raise CoordinateDataError("EOP table is empty")
        self._records = tuple(sorted(records, key=lambda record: record.mjd))
        self._mjds = np.array([record.mjd for record in self._records], dtype=float)

    @classmethod
    def from_file(cls, path: str | Path) -> EopFile:
        records: list[EopRecord] = []
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            tokens = line.split()
            if len(tokens) < 8 or not (tokens[0].isdigit() and tokens[1].isdigit()):
                continue
            records.append(
                EopRecord(
                    year=int(tokens[0]),
                    month=int(tokens[1]),
                    day=int(tokens[2]),
                    mjd=float(tokens[3]),
                    x_arcsec=float(tokens[4]),
                    y_arcsec=float(tokens[5]),
                    ut1_utc=float(tokens[6]),
                    lod=float(tokens[7]),
                )
            )
        return cls(tuple(records))

    @property
    def start_mjd(self) -> float:
        return float(self._mjds[0])

    @property
    def end_mjd(self) -> float:
        return float(self._mjds[-1])

    def at_utc_mjd(self, utc_mjd: float, *, extrapolation: str = "raise") -> EopSample:
        """按 GMAT 语义查询 UTC MJD 时刻的 EOP。

        ``x``/``y`` 和 ``UT1-UTC`` 线性插值，``LOD`` 使用左侧记录值。
        """
        if utc_mjd < self.start_mjd or utc_mjd > self.end_mjd:
            if extrapolation != "clamp":
                raise CoordinateDataError(
                    f"EOP epoch {utc_mjd} outside fixture range {self.start_mjd}..{self.end_mjd}"
                )
            utc_mjd = min(max(utc_mjd, self.start_mjd), self.end_mjd)

        right_index = int(np.searchsorted(self._mjds, utc_mjd, side="right"))
        if right_index == 0:
            left = right = self._records[0]
        elif right_index >= len(self._records):
            left = right = self._records[-1]
        else:
            left = self._records[right_index - 1]
            right = self._records[right_index]

        if left.mjd != right.mjd and right.mjd - left.mjd > 1.5:
            raise CoordinateDataError(
                f"EOP epoch {utc_mjd} falls in an uncovered fixture gap "
                f"between {left.mjd} and {right.mjd}"
            )

        ratio = 0.0 if left.mjd == right.mjd else (utc_mjd - left.mjd) / (right.mjd - left.mjd)

        x_arcsec = left.x_arcsec + ratio * (right.x_arcsec - left.x_arcsec)
        y_arcsec = left.y_arcsec + ratio * (right.y_arcsec - left.y_arcsec)
        ut1_utc = left.ut1_utc + ratio * (right.ut1_utc - left.ut1_utc)
        return EopSample(
            mjd=utc_mjd,
            x_rad=x_arcsec * ARCSEC_TO_RAD,
            y_rad=y_arcsec * ARCSEC_TO_RAD,
            ut1_utc=ut1_utc,
            lod=left.lod,
        )
