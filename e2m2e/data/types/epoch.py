"""时间类型：UTC/TDB/TAI 等时间尺度的类型别名与约定。

**TDB 作动力学统一时间**（ADR 0015）：算法层/数值层内部统一用 ET(TDB) 或
JD_TDB；只有接口边界（api/Pydantic/输出格式）才转 UTC。

时间转换是 ``EphemerisProvider`` 的方法（不单独 TimeSystem 类）：utc_to_tdb、
et_to_utc、utc_to_tai、tai_to_tt、tt_to_tdb、jd_tdb_to_et。

实现状态：骨架。类型别名已可用，转换待 ``EphemerisProvider`` 落定。
"""

from __future__ import annotations

from typing import TypeAlias

__all__ = ["Epoch", "EpochUtc", "EtSec", "JdTdb"]

#: 历元时间（通用别名，时间尺度由调用上下文约定）。
Epoch: TypeAlias = float | str

#: UTC 历元（ISO 字符串或 ``[年,月,日,时,分,秒]`` 序列）。
EpochUtc: TypeAlias = str | list[float] | tuple[float, ...]

#: SPICE ET 秒（TDB 时间尺度）。
EtSec: TypeAlias = float

#: 儒略日（TDB 时间尺度）。
JdTdb: TypeAlias = float
