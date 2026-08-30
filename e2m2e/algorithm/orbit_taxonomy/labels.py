"""轨道分类学 42 标签词汇（orbit taxonomy labels）。

三层分类学（issue #581，ADR 0042）：平动点轨道 27 + 月心轨道 4 + 共振
轨道 11，标签措辞逐字采自 STK 未发布 CODE（Cislunar Orbit Designer）
组件；判定判据无公开出处，为 e2m2e 自定解析判据（见 classify 模块）。

命名约定：

- 规范字符串为 snake_case，如 ``halo_l2_northern``、``low_prograde_eastern``；
  它是序列化键（MCP 响应、catalog 记录），结构化字段是语义载荷。
- 共振比 p:q = 卫星:月球：卫星 p 圈 per 月球 q 圈，T/T☾ = q/p
  （2:1 内共振、1:2 外共振），与 spatiography 共振梯子的 k:k_b 同向。
- 南北（northern/southern）由轨迹 z 极值符号定义，东西
  （eastern/western）由月心会合系近月点方向定义（ADR 0042 判据）。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

__all__ = [
    "TAXONOMY",
    "TAXONOMY_BY_CANONICAL",
    "Hemisphere",
    "TaxonomyCategory",
    "TaxonomyLabel",
    "label_legend",
    "parse_taxonomy_label",
]


class TaxonomyCategory(enum.Enum):
    """标签顶层类别。"""

    LIBRATION_POINT = "libration_point"
    MOON_CENTERED = "moon_centered"
    RESONANT = "resonant"


class Hemisphere(enum.Enum):
    """半球/朝向修饰词。"""

    NORTHERN = "northern"
    SOUTHERN = "southern"
    EASTERN = "eastern"
    WESTERN = "western"


@dataclass(frozen=True)
class TaxonomyLabel:
    """单个分类标签：类别 + 族 + 平动点/半球/共振比修饰。

    ``canonical`` 是序列化键；结构化字段描述语义（catalog 记录与 MCP
    响应只落规范字符串，结构经 :func:`label_legend` 或
    :func:`parse_taxonomy_label` 还原）。
    """

    category: TaxonomyCategory
    family: str
    libration_point: int | None = None
    hemisphere: Hemisphere | None = None
    resonance: tuple[int, int] | None = None

    @property
    def canonical(self) -> str:
        parts = [self.family]
        if self.resonance is not None:
            parts.append(f"{self.resonance[0]}_{self.resonance[1]}")
        elif self.libration_point is not None:
            parts.append(f"l{self.libration_point}")
        if self.hemisphere is not None:
            parts.append(self.hemisphere.value)
        return "_".join(parts)


def _collinear_l(name: str, family: str) -> TaxonomyLabel:
    point = int(name[-1])
    return TaxonomyLabel(TaxonomyCategory.LIBRATION_POINT, family, libration_point=point)


def _hemis_pair() -> tuple[Hemisphere, Hemisphere]:
    return (Hemisphere.NORTHERN, Hemisphere.SOUTHERN)


def _hemispheric(family: str, point: int | None, hemisphere: Hemisphere) -> TaxonomyLabel:
    return TaxonomyLabel(
        TaxonomyCategory.LIBRATION_POINT, family, libration_point=point, hemisphere=hemisphere
    )


#: 42 标签全集，顺序即 issue #581 清单顺序。
TAXONOMY: tuple[TaxonomyLabel, ...] = tuple(
    [
        # 平动点轨道（Libration Point Orbits）
        *(_collinear_l(f"lyapunov_l{i}", "lyapunov") for i in (1, 2, 3)),
        *(
            _hemispheric("halo", point, hemisphere)
            for point in (1, 2, 3)
            for hemisphere in (Hemisphere.NORTHERN, Hemisphere.SOUTHERN)
        ),
        *(_collinear_l(f"axial_l{i}", "axial") for i in (1, 2, 3, 4, 5)),
        *(_collinear_l(f"vertical_l{i}", "vertical") for i in (1, 2, 3, 4, 5)),
        *(_collinear_l(f"longperiod_l{i}", "longperiod") for i in (4, 5)),
        *(_collinear_l(f"shortperiod_l{i}", "shortperiod") for i in (4, 5)),
        *(_hemispheric("butterfly", None, hemisphere) for hemisphere in _hemis_pair()),
        *(_hemispheric("dragonfly", None, hemisphere) for hemisphere in _hemis_pair()),
        # 月球中心轨道（Moon Centered Orbits）
        TaxonomyLabel(TaxonomyCategory.MOON_CENTERED, "distant_retrograde"),
        TaxonomyLabel(TaxonomyCategory.MOON_CENTERED, "distant_prograde"),
        TaxonomyLabel(
            TaxonomyCategory.MOON_CENTERED,
            "low_prograde",
            hemisphere=Hemisphere.EASTERN,
        ),
        TaxonomyLabel(
            TaxonomyCategory.MOON_CENTERED,
            "low_prograde",
            hemisphere=Hemisphere.WESTERN,
        ),
        # 共振轨道（Resonant Orbits），p:q = 卫星:月球
        *(
            TaxonomyLabel(TaxonomyCategory.RESONANT, "resonant", resonance=(p, q))
            for (p, q) in (
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 1),
                (3, 1),
                (3, 2),
                (3, 4),
                (2, 3),
                (4, 1),
                (4, 3),
            )
        ),
    ]
)

TAXONOMY_BY_CANONICAL: dict[str, TaxonomyLabel] = {label.canonical: label for label in TAXONOMY}


def parse_taxonomy_label(canonical: str) -> TaxonomyLabel:
    """规范字符串 → 标签；不在词汇表内时抛 ``ValueError``。"""
    try:
        return TAXONOMY_BY_CANONICAL[canonical]
    except KeyError as exc:
        raise ValueError(f"未知的轨道分类学标签：{canonical!r}") from exc


def label_legend() -> dict[str, dict[str, object]]:
    """规范字符串 → 结构化字段的图例（MCP legend / 文档用）。"""
    legend: dict[str, dict[str, object]] = {}
    for label in TAXONOMY:
        legend[label.canonical] = {
            "category": label.category.value,
            "family": label.family,
            "libration_point": label.libration_point,
            "hemisphere": None if label.hemisphere is None else label.hemisphere.value,
            "resonance_p": None if label.resonance is None else label.resonance[0],
            "resonance_q": None if label.resonance is None else label.resonance[1],
        }
    return legend
