"""orbit_taxonomy 标签词汇测试：42 标签全集、规范字符串、解析与图例。"""

from __future__ import annotations

import pytest

from e2m2e.data.catalog.terminology import (  # ADR 0044：词表已迁数据层
    TAXONOMY,
    TAXONOMY_BY_CANONICAL,
    Hemisphere,
    TaxonomyCategory,
    label_legend,
    parse_taxonomy_label,
)

pytestmark = pytest.mark.theory

#: issue #581 清单的规范字符串全集（逐字对照，顺序即清单顺序）。
EXPECTED_CANONICAL = (
    # 平动点轨道
    "lyapunov_l1",
    "lyapunov_l2",
    "lyapunov_l3",
    "halo_l1_northern",
    "halo_l1_southern",
    "halo_l2_northern",
    "halo_l2_southern",
    "halo_l3_northern",
    "halo_l3_southern",
    "axial_l1",
    "axial_l2",
    "axial_l3",
    "axial_l4",
    "axial_l5",
    "vertical_l1",
    "vertical_l2",
    "vertical_l3",
    "vertical_l4",
    "vertical_l5",
    "longperiod_l4",
    "longperiod_l5",
    "shortperiod_l4",
    "shortperiod_l5",
    "butterfly_northern",
    "butterfly_southern",
    "dragonfly_northern",
    "dragonfly_southern",
    # 月球中心轨道
    "distant_retrograde",
    "distant_prograde",
    "low_prograde_eastern",
    "low_prograde_western",
    # 共振轨道（p:q = 卫星:月球）
    "resonant_1_1",
    "resonant_1_2",
    "resonant_1_3",
    "resonant_1_4",
    "resonant_2_1",
    "resonant_3_1",
    "resonant_3_2",
    "resonant_3_4",
    "resonant_2_3",
    "resonant_4_1",
    "resonant_4_3",
)


def test_taxonomy_covers_issue_list_verbatim():
    """42 标签与 issue 清单一一对应，顺序一致。"""
    assert [label.canonical for label in TAXONOMY] == list(EXPECTED_CANONICAL)
    assert len(TAXONOMY) == 42
    assert len(TAXONOMY_BY_CANONICAL) == 42


def test_category_counts():
    """三大类计数：平动点 27 / 月心 4 / 共振 11。"""
    counts = {category: 0 for category in TaxonomyCategory}
    for label in TAXONOMY:
        counts[label.category] += 1
    assert counts == {
        TaxonomyCategory.LIBRATION_POINT: 27,
        TaxonomyCategory.MOON_CENTERED: 4,
        TaxonomyCategory.RESONANT: 11,
    }


@pytest.mark.parametrize("canonical", EXPECTED_CANONICAL)
def test_parse_round_trip(canonical: str):
    """规范字符串解析还原同一标签（canonical 幂等）。"""
    label = parse_taxonomy_label(canonical)
    assert label.canonical == canonical
    assert TAXONOMY_BY_CANONICAL[canonical] is label


def test_parse_rejects_unknown():
    with pytest.raises(ValueError, match="未知的轨道分类学标签"):
        parse_taxonomy_label("halo_l7_northern")


def test_structured_fields():
    """结构化字段的语义载荷抽查。"""
    halo = parse_taxonomy_label("halo_l2_southern")
    assert halo.category is TaxonomyCategory.LIBRATION_POINT
    assert halo.family == "halo"
    assert halo.libration_point == 2
    assert halo.hemisphere is Hemisphere.SOUTHERN
    assert halo.resonance is None

    low = parse_taxonomy_label("low_prograde_eastern")
    assert low.category is TaxonomyCategory.MOON_CENTERED
    assert low.hemisphere is Hemisphere.EASTERN

    res = parse_taxonomy_label("resonant_3_4")
    assert res.category is TaxonomyCategory.RESONANT
    assert res.resonance == (3, 4)

    butterfly = parse_taxonomy_label("butterfly_northern")
    assert butterfly.libration_point is None
    assert butterfly.hemisphere is Hemisphere.NORTHERN


def test_legend_shape():
    """图例每个条目带六个结构化键，键集与 ADR 0042 一致。"""
    legend = label_legend()
    assert len(legend) == 42
    for entry in legend.values():
        assert set(entry) == {
            "category",
            "family",
            "libration_point",
            "hemisphere",
            "resonance_p",
            "resonance_q",
        }
    assert legend["resonant_2_1"]["resonance_p"] == 2
    assert legend["resonant_2_1"]["resonance_q"] == 1
    assert legend["distant_retrograde"]["category"] == "moon_centered"
