"""术语清单数据层测试（ADR 0044）：42 标签表、图例结构与闭值集常量。

标签表自 algorithm/orbit_taxonomy 迁来（决策 2），本文件锁定其结构
完整性与三份闭值集常量的存在；跨层同步（ingest 映射像 = 族名闭值集）
在 tests/api/test_catalog_terminology.py 锁定。
"""

from __future__ import annotations

import pytest

from e2m2e.data.catalog.terminology import (
    RECORD_ORBIT_FAMILIES,
    TAXONOMY,
    TAXONOMY_BY_CANONICAL,
    TRANSFER_TYPES,
    Hemisphere,
    TaxonomyCategory,
    label_legend,
    parse_taxonomy_label,
)

pytestmark = pytest.mark.data


class TestTaxonomyTable:
    def test_forty_two_unique_canonical_labels(self):
        assert len(TAXONOMY) == 42
        assert len(TAXONOMY_BY_CANONICAL) == 42

    def test_category_composition(self):
        categories = [label.category for label in TAXONOMY]
        assert categories.count(TaxonomyCategory.LIBRATION_POINT) == 27
        assert categories.count(TaxonomyCategory.MOON_CENTERED) == 4
        assert categories.count(TaxonomyCategory.RESONANT) == 11

    def test_parse_roundtrip(self):
        for label in TAXONOMY:
            assert parse_taxonomy_label(label.canonical) is label

    def test_parse_unknown_raises(self):
        with pytest.raises(ValueError, match="未知的轨道分类学标签"):
            parse_taxonomy_label("halo_l2_eastern")


class TestLabelLegend:
    def test_legend_covers_all_labels_with_structured_fields(self):
        legend = label_legend()
        assert set(legend) == set(TAXONOMY_BY_CANONICAL)
        for fields in legend.values():
            assert fields["category"] in {"libration_point", "moon_centered", "resonant"}
            assert isinstance(fields["family"], str) and fields["family"]
            assert fields["libration_point"] is None or fields["libration_point"] in (1, 2, 3, 4, 5)
            assert fields["hemisphere"] in {None, *(h.value for h in Hemisphere)}
            if fields["resonance_p"] is None:
                assert fields["resonance_q"] is None
            else:
                assert isinstance(fields["resonance_p"], int)
                assert isinstance(fields["resonance_q"], int)


class TestClosedValueSets:
    def test_record_orbit_families_is_closed_and_sorted(self):
        assert tuple(sorted(set(RECORD_ORBIT_FAMILIES))) == RECORD_ORBIT_FAMILIES
        assert "halo" in RECORD_ORBIT_FAMILIES and "dro" in RECORD_ORBIT_FAMILIES

    def test_transfer_types(self):
        assert set(TRANSFER_TYPES) == {"HMN", "LGA", "WSB", "low_thrust"}
