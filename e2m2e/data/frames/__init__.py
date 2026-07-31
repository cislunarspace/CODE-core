"""时空参考系数据：EOP、闰秒表、历表句柄。

只留**数据**（EOP 文件、闰秒表、r2s2/SPICE 句柄管理）；转换**算法**在
``algorithm/coordinate/``（强化现有 Axes/Origin/CoordinateSystem 抽象，不新增
Frame 抽象，见 ADR 0015）。

- ``eop.py``：EOP 文件解析（源 ``core/coordinate/gmat_eop.py``）。
- ``leap_seconds.py``：闰秒表（源 ``core/coordinate/gmat_eop.py`` 闰秒部分）。
- ``r2s2.py``：r2s2 句柄管理（源 ``core/coordinate/gcrs_ebcrs.py`` 句柄部分）。
- ``spice_frames.py``：SPICE 帧旋转查询。
"""

from .eop import ARCSEC_TO_RAD, JD_MJD_OFFSET, CoordinateDataError, EopFile, EopRecord, EopSample
from .leap_seconds import SECONDS_PER_DAY, LeapSecondRecord, TaiUtcTable
from .r2s2 import R2S2Adapter
from .spice_frames import frame_rotation

__all__ = [
    "ARCSEC_TO_RAD",
    "JD_MJD_OFFSET",
    "CoordinateDataError",
    "EopFile",
    "EopRecord",
    "EopSample",
    "SECONDS_PER_DAY",
    "LeapSecondRecord",
    "TaiUtcTable",
    "R2S2Adapter",
    "frame_rotation",
]
