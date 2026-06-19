"""GMAT 闰秒与地球定向参数数据读取模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gmat_data import CoordinateDataError

SECONDS_PER_DAY = 86400.0
ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
JD_MJD_OFFSET = 2400000.5


@dataclass(frozen=True)
class LeapSecondRecord:
    julian_date: float
    offset1: float
    offset2: float
    offset3: float


class TaiUtcTable:
    """GMAT ``tai-utc.dat`` reader."""

    def __init__(self, records: tuple[LeapSecondRecord, ...]) -> None:
        if not records:
            raise CoordinateDataError("TAI-UTC table is empty")
        self._records = tuple(sorted(records, key=lambda record: record.julian_date))

    @classmethod
    def from_file(cls, path: str | Path) -> TaiUtcTable:
        records: list[LeapSecondRecord] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            tokens = line.replace("(", " ( ").replace(")", " ) ").split()
            julian_date = float(tokens[4])
            offset1 = float(tokens[6])
            offset2 = float(tokens[12].rstrip("."))
            offset3 = float(tokens[15].removesuffix("S"))
            records.append(LeapSecondRecord(julian_date, offset1, offset2, offset3))
        return cls(tuple(records))

    def tai_minus_utc(self, utc_mjd: float) -> float:
        """返回 UTC MJD 时刻的 ``TAI-UTC`` 秒数。"""
        julian_date = utc_mjd + JD_MJD_OFFSET
        selected = self._records[0]
        for record in self._records:
            if julian_date >= record.julian_date:
                selected = record
            else:
                break
        return selected.offset1 + (utc_mjd - selected.offset2) * selected.offset3

    def utc_to_tai_mjd(self, utc_mjd: float) -> float:
        """将 UTC MJD 转为 TAI MJD。"""
        return utc_mjd + self.tai_minus_utc(utc_mjd) / SECONDS_PER_DAY

    def tai_to_utc_mjd(self, tai_mjd: float) -> float:
        """将 TAI MJD 转为 UTC MJD。"""
        utc_mjd = tai_mjd - self.tai_minus_utc(tai_mjd) / SECONDS_PER_DAY
        for _ in range(3):
            utc_mjd = tai_mjd - self.tai_minus_utc(utc_mjd) / SECONDS_PER_DAY
        return utc_mjd


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
    """GMAT C04 EOP reader.

    GMAT 的现代 C04 路径只消费 ``mjd, x, y, UT1-UTC, LOD``；后续
    ``dPsi/dEps`` 字段保留在原文件中但不进入当前 native ITRF 链路。
    """

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
