"""Primer 常数集（spatiography 专用，Rosengren et al. 2026 §5 自洽集合）。

论文《The Astrodynamics Primer on Cislunar and Translunar Space》的分区尺度
（Laplace 半径、Hill 球、SOI 族、共振梯、tidal parity）全部以下列常数复现。
这些数值构成一套**自洽的已发表集合**：GM 取 JPL/NAIF SPICE，月球平均轨道
根数取 Simon et al. 1994，$J_2$ 与参考半径取 GGM02 归一。

与 ``universal``/``datum`` 的取值刻意不同（例如月球平均半长轴 383397.7725 km
是轨道根数意义下的量，不同于 DE421 特征长度 384400 km）——分区计算必须整套
使用本节常数，不得与其他来源混搭，否则黄金值系统性漂移。见 ADR 0041。
"""

from __future__ import annotations

from typing import cast

from ._loader import _load_section
from .sources import ConstantSource

_PRIMER: dict[str, dict[str, object]] = cast(dict[str, dict[str, object]], _load_section("primer"))


def _value(key: str) -> float:
    try:
        entry = _PRIMER[key]
    except KeyError as exc:
        raise KeyError(f"constants.toml 中 [primer] 缺少 {key!r}") from exc
    if isinstance(entry, dict):
        return float(entry["value"])  # type: ignore[arg-type]
    return float(entry)  # type: ignore[arg-type]


def _source(key: str) -> ConstantSource:
    entry = _PRIMER[key]
    if isinstance(entry, dict):
        return ConstantSource(entry.get("source", "NAIF"))
    return ConstantSource.NAIF


# 地球二阶带谐系数 J2。GGM02 重力场归一（Tapley et al. 2005）。
PRIMER_EARTH_J2: float = _value("earth_j2")
PRIMER_EARTH_J2_SOURCE: ConstantSource = _source("earth_j2")

# 月球二阶带谐系数 J2。Legnaro & Efthymiopoulos 2024 采用值。
PRIMER_MOON_J2: float = _value("moon_j2")
PRIMER_MOON_J2_SOURCE: ConstantSource = _source("moon_j2")

# 月球平均轨道半长轴（km）。Simon et al. 1994。
PRIMER_MOON_A_KM: float = _value("moon_a_km")
PRIMER_MOON_A_KM_SOURCE: ConstantSource = _source("moon_a_km")

# 月球平均轨道偏心率。Simon et al. 1994。
PRIMER_MOON_ECC: float = _value("moon_ecc")
PRIMER_MOON_ECC_SOURCE: ConstantSource = _source("moon_ecc")

# 月球平均轨道倾角（deg，相对黄道）。Simon et al. 1994。
PRIMER_MOON_INC_DEG: float = _value("moon_inc_deg")
PRIMER_MOON_INC_DEG_SOURCE: ConstantSource = _source("moon_inc_deg")

# 太阳视地心轨道半长轴（au）。Simon et al. 1994。
PRIMER_SUN_A_AU: float = _value("sun_a_au")
PRIMER_SUN_A_AU_SOURCE: ConstantSource = _source("sun_a_au")

# 太阳视地心轨道偏心率。Simon et al. 1994。
PRIMER_SUN_ECC: float = _value("sun_ecc")
PRIMER_SUN_ECC_SOURCE: ConstantSource = _source("sun_ecc")

# 地球引力参数 GM⊕（km³/s²）。JPL/NAIF SPICE 内核值（论文 §5 引用）。
PRIMER_EARTH_GM: float = _value("earth_gm")
PRIMER_EARTH_GM_SOURCE: ConstantSource = _source("earth_gm")

# 月球引力参数 GM☾（km³/s²）。JPL/NAIF SPICE 内核值（论文 §5 引用）。
PRIMER_MOON_GM: float = _value("moon_gm")
PRIMER_MOON_GM_SOURCE: ConstantSource = _source("moon_gm")

# 太阳引力参数 GM☉（km³/s²）。JPL/NAIF SPICE 内核值（论文 §5 引用）。
PRIMER_SUN_GM: float = _value("sun_gm")
PRIMER_SUN_GM_SOURCE: ConstantSource = _source("sun_gm")

# 地球参考半径 R⊕（km）。GGM02 归一（论文 §5 采用，Tapley et al. 2005）。
PRIMER_EARTH_REF_RADIUS_KM: float = _value("earth_ref_radius_km")
PRIMER_EARTH_REF_RADIUS_KM_SOURCE: ConstantSource = _source("earth_ref_radius_km")

# 地球恒星日（s）。IERS 平恒星日，供 GEO 半径等派生量使用。
PRIMER_EARTH_SIDEREAL_DAY_S: float = _value("earth_sidereal_day_s")
PRIMER_EARTH_SIDEREAL_DAY_S_SOURCE: ConstantSource = _source("earth_sidereal_day_s")

# 常数集出处（用于工具响应与文档引用）。
PRIMER_CITATION = (
    "Rosengren et al. 2026, The Astrodynamics Primer on Cislunar and "
    "Translunar Space, §5 (GM: JPL/NAIF SPICE; lunar elements: Simon et al. "
    "1994; J2/R⊕: GGM02)"
)
