"""catalog_terminology 出口测试（ADR 0044）：响应内容、注册与跨层同步。

三份闭值集的单一来源在数据层；本文件锁（a）出口响应与数据层同源、
（b）工具已注册（ADR 0043 决策 6 准入判据的首例执行）、（c）ingest
映射像 = 族名闭值集、转移类型闭值集 = 算法层 state_frame 派生键——
跨层词汇表不得各自漂移。
"""

from __future__ import annotations

import pytest

from e2m2e.api.facade import Facade, mcp_tools, tool_inventory
from e2m2e.data.catalog.terminology import (
    RECORD_ORBIT_FAMILIES,
    TRANSFER_TYPES,
    label_legend,
)

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


class TestCrossLayerSync:
    def test_ingest_family_image_equals_closed_set(self):
        """ingest 映射像（含小写生成器类型）= 族名闭值集，双向不漂移。"""
        from e2m2e.api import catalog_ingest

        image = {family for family, _point in catalog_ingest._DESIGN_FAMILY_POINT.values()}
        image |= {
            selection.lower()
            for selection in ("HALO", "NRHO", "AXIAL", "LISSAJOUS")
            if selection not in catalog_ingest._DESIGN_FAMILY_POINT
        }
        assert image == set(RECORD_ORBIT_FAMILIES)

    def test_transfer_types_match_state_frame_derivation_keys(self):
        from e2m2e.algorithm.transfer import _STATE_FRAME_BY_TRANSFER_TYPE

        assert set(TRANSFER_TYPES) == set(_STATE_FRAME_BY_TRANSFER_TYPE)


def test_query_request_declares_family_filter():
    from e2m2e.api.models import CatalogQueryRequest

    assert "family_id" in CatalogQueryRequest.model_fields
