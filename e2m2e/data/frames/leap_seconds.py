"""闰秒表（TAI−UTC）文件解析。

数据层（ADR 0011 迁移，源：``core/coordinate/gmat_eop.py`` 的闰秒部分）。
只留**数据解析**；时间尺度转换（UTC→TAI→TT→TDB）在
``EphemerisProvider`` 与 ``algorithm/coordinate/``。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .eop import JD_MJD_OFFSET, CoordinateDataError

SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class LeapSecondRecord:
    julian_date: float
    offset1: float
    offset2: float
    offset3: float


class TaiUtcTable:
    """GMAT ``tai-utc.dat`` reader。"""

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
