"""GMAT 闰秒与地球定向参数数据 shim（ADR 0011 迁移）。

实现已迁至数据层：``e2m2e.data.frames.eop``（EOP 文件解析）与
``e2m2e.data.frames.leap_seconds``（闰秒表），旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.data.frames.eop import (
    ARCSEC_TO_RAD,
    JD_MJD_OFFSET,
    CoordinateDataError,
    EopFile,
    EopRecord,
    EopSample,
)
from e2m2e.data.frames.leap_seconds import SECONDS_PER_DAY, LeapSecondRecord, TaiUtcTable

__all__ = [
    "SECONDS_PER_DAY",
    "ARCSEC_TO_RAD",
    "JD_MJD_OFFSET",
    "CoordinateDataError",
    "LeapSecondRecord",
    "TaiUtcTable",
    "EopRecord",
    "EopSample",
    "EopFile",
]
