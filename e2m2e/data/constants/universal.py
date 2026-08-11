"""通用物理常量（全库一套）。

单位统一采用 km / s / kg 系；来源标注在 constants.toml。
数值由仓库根 ``constants.toml`` 单一来源加载。
"""

from __future__ import annotations

from typing import cast

from ._loader import _load_section
from .sources import ConstantSource

_UNIVERSAL: dict[str, dict[str, object]] = cast(
    dict[str, dict[str, object]], _load_section("universal")
)


def _value(key: str) -> float:
    try:
        entry = _UNIVERSAL[key]
    except KeyError as exc:
        raise KeyError(f"constants.toml 中 [universal] 缺少 {key!r}") from exc
    if isinstance(entry, dict):
        return float(entry["value"])  # type: ignore[arg-type]
    return float(entry)  # type: ignore[arg-type]


def _source(key: str) -> ConstantSource:
    entry = _UNIVERSAL[key]
    if isinstance(entry, dict):
        return ConstantSource(entry.get("source", "SI"))
    return ConstantSource.SI


# 真空中的光速（km/s）。SI 定义值，精确。
SPEED_OF_LIGHT_KMS: float = _value("speed_of_light_kms")
SPEED_OF_LIGHT_KMS_SOURCE: ConstantSource = _source("speed_of_light_kms")

# 万有引力常数 G（km³ / (kg · s²)）。CODATA 2018 推荐值。
GRAVITATIONAL_CONSTANT: float = _value("gravitational_constant")
GRAVITATIONAL_CONSTANT_SOURCE: ConstantSource = _source("gravitational_constant")

# 天文单位（km）。IAU 2012 决议 B2。
AU_KM: float = _value("au_km")
AU_KM_SOURCE: ConstantSource = _source("au_km")

# 一天的秒数。定义值。
SECONDS_PER_DAY: float = _value("seconds_per_day")
SECONDS_PER_DAY_SOURCE: ConstantSource = _source("seconds_per_day")

# 儒略年的天数。定义值。
DAYS_PER_JULIAN_YEAR: float = _value("days_per_julian_year")
DAYS_PER_JULIAN_YEAR_SOURCE: ConstantSource = _source("days_per_julian_year")

# 儒略世纪的天数。定义值。
DAYS_PER_JULIAN_CENTURY: float = _value("days_per_julian_century")
DAYS_PER_JULIAN_CENTURY_SOURCE: ConstantSource = _source("days_per_julian_century")

# 儒略年的秒数。由定义派生。
SECONDS_PER_JULIAN_YEAR: float = DAYS_PER_JULIAN_YEAR * SECONDS_PER_DAY

# 地月平均距离（km）。Cui et al. 2025。
EARTH_MOON_DISTANCE_KM: float = _value("earth_moon_distance_km")
EARTH_MOON_DISTANCE_KM_SOURCE: ConstantSource = _source("earth_moon_distance_km")

# 千米 → 米换算因子。定义值。
KM_TO_M: float = _value("km_to_m")
KM_TO_M_SOURCE: ConstantSource = _source("km_to_m")

# 太阳辐射通量（W/m²）。工程默认值，GMAT/IERS 1996 兼容。
SOLAR_FLUX_W_M2: float = _value("solar_flux_w_m2")
SOLAR_FLUX_W_M2_SOURCE: ConstantSource = _source("solar_flux_w_m2")

# 太阳辐射通量（W/m²）。现代 TSI 备选值（Pesce 2023）。
SOLAR_FLUX_TSI_W_M2: float = _value("solar_flux_tsi_w_m2")
SOLAR_FLUX_TSI_W_M2_SOURCE: ConstantSource = _source("solar_flux_tsi_w_m2")

# 1 AU 处太阳辐射压（N/m²）。由 SOLAR_FLUX_W_M2 / (c · 1000) 派生。
SOLAR_PRESSURE_1AU: float = SOLAR_FLUX_W_M2 / (SPEED_OF_LIGHT_KMS * KM_TO_M)
