"""天体参数目录。

按天体组织，每个天体提供各基准下的 GM、半径、扁率、NAIF ID、自转角速度等。
半径区分两种语义：
- ``mean_radius_km``：平均/赤道半径，用于阴影、SRP、相对论等几何计算；
- ``gravity_ref_radius_km``：重力场参考半径，来自系数文件头，用于球谐、
  固潮等重力模型。

数值由仓库根 ``constants.toml`` 单一来源加载。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from ._loader import _load_section
from .sources import ConstantSource

_BODIES: dict[str, dict[str, object]] = cast(dict[str, dict[str, object]], _load_section("body"))


def _scalar(section: dict[str, object], key: str) -> float | None:
    entry = section.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return float(entry["value"])  # type: ignore[arg-type]
    return float(entry)  # type: ignore[arg-type]


def _source(section: dict[str, object], key: str) -> ConstantSource | None:
    entry = section.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return ConstantSource(entry.get("source", "SI"))
    return ConstantSource.SI


def _int_value(section: dict[str, object], key: str) -> int | None:
    entry = section.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return int(entry["value"])  # type: ignore[call-overload]
    return int(entry)  # type: ignore[call-overload]


def _gm_table(section: dict[str, object]) -> dict[str, float]:
    gm = section.get("gm")
    if not isinstance(gm, dict):
        return {}
    return {
        datum: float(entry["value"]) if isinstance(entry, dict) else float(entry)
        for datum, entry in gm.items()
    }


def _gm_sources(section: dict[str, object]) -> dict[str, ConstantSource]:
    gm = section.get("gm")
    if not isinstance(gm, dict):
        return {}
    return {
        datum: ConstantSource(entry.get("source", datum))
        if isinstance(entry, dict)
        else ConstantSource(datum)
        for datum, entry in gm.items()
    }


@dataclass(frozen=True)
class Body:
    """天体参数。

    ``gm_by_datum`` 为各基准下的引力参数（km³/s²）；``mean_radius_km`` 为
    几何/制图平均半径；``gravity_ref_radius_km`` 为重力场系数文件给出的
    参考半径；``naif_id`` 为 SPICE/NAIF 整数 ID；``rotation_rate_rad_s`` 为
    自转角速度（rad/s），按场景分别命名。
    """

    name: str
    gm_by_datum: dict[str, float] = field(default_factory=dict)
    gm_sources: dict[str, ConstantSource] = field(default_factory=dict)
    mean_radius_km: float | None = None
    mean_radius_source: ConstantSource | None = None
    gravity_ref_radius_km: float | None = None
    gravity_ref_radius_source: ConstantSource | None = None
    flattening: float | None = None
    flattening_source: ConstantSource | None = None
    naif_id: int | None = None
    rotation_rate_iers_rad_s: float | None = None
    rotation_rate_iers_source: ConstantSource | None = None
    rotation_rate_gmat_rad_s: float | None = None
    rotation_rate_gmat_source: ConstantSource | None = None

    def require_mean_radius_km(self) -> float:
        """平均半径（km）；该天体未定义半径时抛错（如 EMB 无半径概念）。"""
        if self.mean_radius_km is None:
            raise KeyError(f"constants.toml 中 [body.{self.name}] 缺少 'mean_radius_km'")
        return self.mean_radius_km


def _build_body(name: str) -> Body:
    section = _BODIES.get(name, {})
    return Body(
        name=name,
        gm_by_datum=_gm_table(section),
        gm_sources=_gm_sources(section),
        mean_radius_km=_scalar(section, "mean_radius_km"),
        mean_radius_source=_source(section, "mean_radius_km"),
        gravity_ref_radius_km=_scalar(section, "gravity_ref_radius_km"),
        gravity_ref_radius_source=_source(section, "gravity_ref_radius_km"),
        flattening=_scalar(section, "flattening"),
        flattening_source=_source(section, "flattening"),
        naif_id=_int_value(section, "naif_id"),
        rotation_rate_iers_rad_s=_scalar(section, "rotation_rate_iers_rad_s"),
        rotation_rate_iers_source=_source(section, "rotation_rate_iers_rad_s"),
        rotation_rate_gmat_rad_s=_scalar(section, "rotation_rate_gmat_rad_s"),
        rotation_rate_gmat_source=_source(section, "rotation_rate_gmat_rad_s"),
    )


#: 太阳。
SUN = _build_body("SUN")

#: 地球。
EARTH = _build_body("EARTH")

#: 月球。
MOON = _build_body("MOON")

#: 地月系质心（Earth-Moon Barycenter）。
EMB = _build_body("EMB")

#: 水星。
MERCURY = _build_body("MERCURY")

#: 金星。
VENUS = _build_body("VENUS")

#: 火星。
MARS = _build_body("MARS")

#: 木星。
JUPITER = _build_body("JUPITER")

#: 土星。
SATURN = _build_body("SATURN")

#: 天王星。
URANUS = _build_body("URANUS")

#: 海王星。
NEPTUNE = _build_body("NEPTUNE")

#: 冥王星。
PLUTO = _build_body("PLUTO")

#: 天体名 → Body 映射（供按名称查询，如 SPICEManager.get_gm）。
_BODIES_BY_NAME: dict[str, Body] = {
    body.name: body
    for body in (
        SUN,
        EARTH,
        MOON,
        EMB,
        MERCURY,
        VENUS,
        MARS,
        JUPITER,
        SATURN,
        URANUS,
        NEPTUNE,
        PLUTO,
    )
}
