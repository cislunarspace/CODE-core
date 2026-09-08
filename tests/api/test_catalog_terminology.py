"""catalog_terminology 出口测试（ADR 0044）：响应内容与工具注册。"""

from __future__ import annotations

import pytest

from e2m2e.api.facade import Facade, mcp_tools, tool_inventory
from e2m2e.data.catalog.terminology import RECORD_ORBIT_FAMILIES, TRANSFER_TYPES, label_legend

pytestmark = pytest.mark.interface


class TestCatalogTerminology:
    def test_response_carries_the_three_lists(self):
        response = Facade().catalog.catalog_terminology()
        assert response.taxonomy_labels == label_legend()
        assert len(response.taxonomy_labels) == 42
        assert response.orbit_families == list(RECORD_ORBIT_FAMILIES)
        assert response.transfer_types == list(TRANSFER_TYPES)

    def test_record_families_cover_every_ingestable_family(self):
        """闭值集 = 映射表像 ∪ 小写生成器类型（ADR 0044 决策 2 的不变式）。

        漂移是静默的（#627 新增 RO 时 ro 就漏在闭值集外）：设计映射表与
        族生成入口两处能盖上库记录的族名必须落在闭值集内。
        """
        from e2m2e.api.catalog_ingest import _DESIGN_FAMILY_POINT
        from e2m2e.api.models import FamilyGenerationRequest

        expected = {family for family, _ in _DESIGN_FAMILY_POINT.values()}
        expected |= {
            orbit_type.lower() for orbit_type, _ in FamilyGenerationRequest.valid_range_contexts()
        }
        assert expected <= set(RECORD_ORBIT_FAMILIES), (
            f"闭值集缺 {sorted(expected - set(RECORD_ORBIT_FAMILIES))}"
        )

    def test_tool_is_registered_on_catalog_class(self):
        # ADR 0043 决策 6 第二款准入：内容被响应字段引用且无既有工具可供给
        facade = Facade()
        assert "catalog_terminology" in set(mcp_tools(facade.catalog))
        assert any(i.name == "catalog_terminology" for i in tool_inventory(facade))
