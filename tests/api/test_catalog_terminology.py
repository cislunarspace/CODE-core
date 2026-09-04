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

    def test_tool_is_registered_on_catalog_class(self):
        # ADR 0043 决策 6 第二款准入：内容被响应字段引用且无既有工具可供给
        facade = Facade()
        assert "catalog_terminology" in set(mcp_tools(facade.catalog))
        assert any(i.name == "catalog_terminology" for i in tool_inventory(facade))
