"""基准集定义。

以"基准（datum）"为一等概念，每套基准内部 GM、μ、特征长度、特征时间
来自同一来源，保持自洽。数值由仓库根 ``constants.toml`` 加载。

基准按字段形状分成三类，类型上精确区分"必有值"与"可选值"：
- ``_FullEarthMoonDatumSpec``：DE421，μ、GM 表、特征长度/时间齐备；
- ``_EarthMoonDatumSpec``：DE440/DE430，μ 与 GM 表必有值，特征尺度未提供
  （避免编造），为合法 ``None``；
- ``_ShapeDatumSpec``：WGS-84，只定义地球形状/地球 GM。

调用方取"必有值"字段时得到 ``float``；取可选字段时类型为 ``float | None``，
需自行确认该基准确实提供该值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from ._loader import _load_section
from .sources import ConstantSource
from .universal import SECONDS_PER_DAY

_DATUMS: dict[str, dict[str, dict[str, object]]] = cast(
    dict[str, dict[str, dict[str, object]]], _load_section("datum")
)


def _datum_value(datum: str, key: str) -> float | None:
    section = _DATUMS.get(datum)
    if section is None:
        return None
    entry = section.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return float(entry["value"])  # type: ignore[arg-type]
    return float(entry)  # type: ignore[arg-type]


def _datum_require(datum: str, key: str) -> float:
    """取必有值的基准字段；缺失即抛错（与 universal._value 同范式）。"""
    value = _datum_value(datum, key)
    if value is None:
        raise KeyError(f"constants.toml 中 [datum.{datum}] 缺少 {key!r}")
    return value


def _datum_source(datum: str, key: str) -> ConstantSource | None:
    section = _DATUMS.get(datum)
    if section is None:
        return None
    entry = section.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return ConstantSource(entry.get("source", datum))
    return ConstantSource(datum)


@dataclass(frozen=True)
class _DatumSpec:
    """基准集公共字段：来源标识与逐字段出处元数据。

    所有 GM 量单位均为 km³/s²；特征长度单位为 km；特征时间单位为 s。
    """

    source: ConstantSource
    field_sources: dict[str, ConstantSource]


@dataclass(frozen=True)
class _EarthMoonDatumSpec(_DatumSpec):
    """地月 CR3BP 基准（DE440/DE430）：μ 与 GM 表必有值，特征尺度未提供。

    特征长度/特征时间由 GM 与星历历元推导，当前阶段未提供，为合法 None。
    """

    # 地月系统质量比（无量纲）。
    mu: float
    # 地球引力参数（km³/s²）。
    earth_gm: float
    # 月球引力参数（km³/s²）。
    moon_gm: float
    # 太阳引力参数（km³/s²）。
    sun_gm: float
    # 地月系质心（EMB）引力参数（km³/s²）。
    emb_gm: float
    # 地月特征长度 l*（km）。
    char_length_km: float | None = None
    # 地月特征时间 t*（s）。
    char_time_s: float | None = None
    # 地球赤道半径（km）。
    earth_radius_km: float | None = None
    # 地球扁率（无量纲）。
    earth_flattening: float | None = None


@dataclass(frozen=True)
class _FullEarthMoonDatumSpec(_EarthMoonDatumSpec):
    """DE421：在 CR3BP 基准基础上，特征长度/时间也必有值。"""

    char_length_km: float
    char_time_s: float


@dataclass(frozen=True)
class _ShapeDatumSpec(_DatumSpec):
    """地球形状基准（WGS-84）：地球 GM、赤道半径、扁率必有值。"""

    # 地球引力参数（km³/s²）。
    earth_gm: float
    # 地球赤道半径（km）。
    earth_radius_km: float
    # 地球扁率（无量纲）。
    earth_flattening: float
    # 地月系统质量比（无量纲）。
    mu: float | None = None
    # 地月特征长度 l*（km）。
    char_length_km: float | None = None
    # 地月特征时间 t*（s）。
    char_time_s: float | None = None
    # 月球引力参数（km³/s²）。
    moon_gm: float | None = None
    # 太阳引力参数（km³/s²）。
    sun_gm: float | None = None
    # 地月系质心（EMB）引力参数（km³/s²）。
    emb_gm: float | None = None


class _DatumEnumMeta(type):
    """支持 ``Datum.DE421.mu`` 形式的命名空间访问。"""

    _members: dict[str, _DatumSpec] = {}

    def __getattr__(cls, name: str) -> _DatumSpec:
        try:
            return cls._members[name]
        except KeyError as exc:
            raise AttributeError(f"Datum 没有名为 {name!r} 的基准") from exc

    def __iter__(cls):
        return iter(cls._members.items())

    def __contains__(cls, name: object) -> bool:
        return name in cls._members


class Datum(metaclass=_DatumEnumMeta):
    """物理常数基准集。

    用法示例::

        from e2m2e.data.constants import Datum
        print(Datum.DE421.mu)
        print(Datum.DE440.earth_gm)
        print(Datum.WGS84.earth_radius_km)
    """

    # 来源：Folta 2022 Table 2；NASA JPL DE421。
    DE421 = _FullEarthMoonDatumSpec(
        source=ConstantSource.DE421,
        mu=_datum_require("DE421", "mu"),
        char_length_km=_datum_require("DE421", "char_length_km"),
        char_time_s=_datum_require("DE421", "char_time_s"),
        earth_gm=_datum_require("DE421", "earth_gm"),
        moon_gm=_datum_require("DE421", "moon_gm"),
        sun_gm=_datum_require("DE421", "sun_gm"),
        emb_gm=_datum_require("DE421", "emb_gm"),
        field_sources={
            "mu": _datum_source("DE421", "mu") or ConstantSource.DE421,
            "char_length_km": _datum_source("DE421", "char_length_km") or ConstantSource.LITERATURE,
            "char_time_s": _datum_source("DE421", "char_time_s") or ConstantSource.LITERATURE,
            "earth_gm": _datum_source("DE421", "earth_gm") or ConstantSource.DE421,
            "moon_gm": _datum_source("DE421", "moon_gm") or ConstantSource.DE421,
            "sun_gm": _datum_source("DE421", "sun_gm") or ConstantSource.DE421,
            "emb_gm": _datum_source("DE421", "emb_gm") or ConstantSource.DE421,
        },
    )

    # 来源：JPL DE440 星历。
    # 特征长度/特征时间由 GM 与星历历元推导，当前阶段未提供，避免编造。
    DE440 = _EarthMoonDatumSpec(
        source=ConstantSource.DE440,
        mu=_datum_require("DE440", "mu"),
        earth_gm=_datum_require("DE440", "earth_gm"),
        moon_gm=_datum_require("DE440", "moon_gm"),
        sun_gm=_datum_require("DE440", "sun_gm"),
        emb_gm=_datum_require("DE440", "emb_gm"),
        field_sources={
            "mu": _datum_source("DE440", "mu") or ConstantSource.DE440,
            "earth_gm": _datum_source("DE440", "earth_gm") or ConstantSource.DE440,
            "moon_gm": _datum_source("DE440", "moon_gm") or ConstantSource.DE440,
            "sun_gm": _datum_source("DE440", "sun_gm") or ConstantSource.DE440,
            "emb_gm": _datum_source("DE440", "emb_gm") or ConstantSource.DE440,
        },
    )

    # 来源：WGS-84 / GMAT R2026a 默认地球形状模型。
    WGS84 = _ShapeDatumSpec(
        source=ConstantSource.WGS84,
        earth_gm=_datum_require("WGS84", "earth_gm"),
        earth_radius_km=_datum_require("WGS84", "earth_radius_km"),
        earth_flattening=_datum_require("WGS84", "earth_flattening"),
        field_sources={
            "earth_gm": _datum_source("WGS84", "earth_gm") or ConstantSource.WGS84,
            "earth_radius_km": _datum_source("WGS84", "earth_radius_km") or ConstantSource.WGS84,
            "earth_flattening": _datum_source("WGS84", "earth_flattening") or ConstantSource.WGS84,
        },
    )

    # DE430 与 DE440 使用相同的 GM 表作为占位；阶段 1 未细分。
    DE430 = DE440


Datum._members = {  # type: ignore[attr-defined]
    "DE421": Datum.DE421,
    "DE430": Datum.DE430,
    "DE440": Datum.DE440,
    "WGS84": Datum.WGS84,
}

# 预计算 SECONDS_PER_DAY 在派生量中可能用到；避免 lint 报告未使用。
_ = SECONDS_PER_DAY
