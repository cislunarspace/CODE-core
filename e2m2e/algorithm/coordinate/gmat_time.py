"""GMAT 兼容时间转换。"""

from __future__ import annotations

import erfa

from ...data.constants import SECONDS_PER_DAY
from ...data.frames.leap_seconds import TaiUtcTable

A1_TAI_OFFSET = 0.0343817
TT_TAI_OFFSET = 32.184
J2000_MJD = 51544.5


class TimeSystemConverter:
    """保持公共 ET 秒输入，同时提供 GMAT 低层时间量。"""

    def __init__(self, tai_utc: TaiUtcTable) -> None:
        self._tai_utc = tai_utc

    def utc_mjd_to_tai_mjd(self, utc_mjd: float) -> float:
        return self._tai_utc.utc_to_tai_mjd(utc_mjd)

    def tai_mjd_to_utc_mjd(self, tai_mjd: float) -> float:
        return self._tai_utc.tai_to_utc_mjd(tai_mjd)

    def utc_mjd_to_tt_mjd(self, utc_mjd: float) -> float:
        return self.utc_mjd_to_tai_mjd(utc_mjd) + TT_TAI_OFFSET / SECONDS_PER_DAY

    def tai_mjd_to_a1_mjd(self, tai_mjd: float) -> float:
        return tai_mjd + A1_TAI_OFFSET / SECONDS_PER_DAY

    def a1_mjd_to_tai_mjd(self, a1_mjd: float) -> float:
        return a1_mjd - A1_TAI_OFFSET / SECONDS_PER_DAY

    def utc_mjd_to_a1_mjd(self, utc_mjd: float) -> float:
        return self.tai_mjd_to_a1_mjd(self.utc_mjd_to_tai_mjd(utc_mjd))

    def a1_mjd_to_utc_mjd(self, a1_mjd: float) -> float:
        return self.tai_mjd_to_utc_mjd(self.a1_mjd_to_tai_mjd(a1_mjd))

    def a1_mjd_to_tt_mjd(self, a1_mjd: float) -> float:
        return self.a1_mjd_to_tai_mjd(a1_mjd) + TT_TAI_OFFSET / SECONDS_PER_DAY

    def et_to_tdb_mjd(self, et: float) -> float:
        return J2000_MJD + et / SECONDS_PER_DAY

    def et_to_tt_mjd(self, et: float) -> float:
        tdb1, tdb2 = self.et_to_tdb_mjd(et), 0.0
        tt1, tt2 = erfa.tdbtt(tdb1, tdb2, 0.0)
        return float(tt1 + tt2)

    def et_to_utc_mjd(self, et: float) -> float:
        tt_mjd = self.et_to_tt_mjd(et)
        tai_mjd = tt_mjd - TT_TAI_OFFSET / SECONDS_PER_DAY
        return self.tai_mjd_to_utc_mjd(tai_mjd)

    def et_to_a1_mjd(self, et: float) -> float:
        tt_mjd = self.et_to_tt_mjd(et)
        tai_mjd = tt_mjd - TT_TAI_OFFSET / SECONDS_PER_DAY
        return self.tai_mjd_to_a1_mjd(tai_mjd)
