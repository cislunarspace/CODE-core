"""valid_ranges 出口测试（ADR 0014 决策 8 请求侧，#620）：键集完整性、
与校验器同源、离散选项。"""

from __future__ import annotations

import pytest

from e2m2e.api.facade import Facade, mcp_tools, tool_inventory
from e2m2e.api.models import (
    DesignOrbitRequest,
    FamilyGenerationRequest,
    RangeSpec,
)
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.interface


class TestValidRangesResponse:
    def test_design_orbit_covers_all_types_and_lissajous_points(self):
        response = Facade().valid_ranges()
        expected_keys = {
            orbit_type if point is None else f"{orbit_type}_L{point}"
            for orbit_type, point in DesignOrbitRequest.valid_range_contexts()
        }
        assert set(response.design_orbit) == expected_keys
        # LISSAJOUS 逐点拆分：L1/L2 共用 7600 km 包络，L3 独立放宽到 100000 km
        assert response.design_orbit["LISSAJOUS_L1"]["amplitude_in"].maximum == 7600.0
        assert response.design_orbit["LISSAJOUS_L2"]["amplitude_in"].maximum == 7600.0
        assert response.design_orbit["LISSAJOUS_L3"]["amplitude_in"].maximum == 100000.0

    def test_design_orbit_ranges_match_validator_source(self):
        response = Facade().valid_ranges()
        expected = DesignOrbitRequest.valid_ranges("HALO")["amplitude"]
        got = response.design_orbit["HALO"]["amplitude"]
        assert isinstance(got, RangeSpec)
        assert (got.minimum, got.maximum) == (expected.minimum, expected.maximum)
        # 开闭语义随源携带：amplitude_out 全局下界为开区间
        global_out = response.design_orbit["DRO"]["amplitude_out"]
        assert global_out.minimum == 0.0
        assert global_out.minimum_inclusive is False

    def test_family_generation_ranges_cover_every_family_and_point(self):
        response = Facade().valid_ranges()
        expected_keys = {
            orbit_type if point is None else f"{orbit_type}_L{point}"
            for orbit_type, point in FamilyGenerationRequest.valid_range_contexts()
        }
        assert set(response.family_generation_ranges) == expected_keys
        # 绑定平动点的族携带 libration_point 允许集（HALO 为 L1/L2，三角族为 L4/L5）；DRO 不带
        halo_point = response.family_generation_ranges["HALO_L1"]["libration_point"]
        assert (halo_point.minimum, halo_point.maximum) == (1, 2)
        spo_point = response.family_generation_ranges["SPO_L5"]["libration_point"]
        assert (spo_point.minimum, spo_point.maximum) == (4, 5)
        assert "libration_point" not in response.family_generation_ranges["DRO"]
        # HALO 振幅上界按平动点折叠点收紧，且排除 0
        halo_l1 = response.family_generation_ranges["HALO_L1"]["max_amplitude_km"]
        assert halo_l1.excluded_values == [0.0]
        assert halo_l1.minimum == -halo_l1.maximum

    def test_family_generation_options_mirror_valid_options(self):
        response = Facade().valid_ranges()
        assert set(response.family_generation_options) == {
            orbit_type for orbit_type, _ in FamilyGenerationRequest.valid_range_contexts()
        }
        assert response.family_generation_options["NRHO"] == {
            "continuation_direction": ["toward-moon"],
            "sampling_mode": ["halo-segment"],
        }

    def test_response_envelope_and_registration(self):
        facade = Facade()
        response = facade.valid_ranges()
        assert response.status is ConvergenceState.CONVERGED
        assert response.cause is FailureCause.NONE
        assert "条件值域" in response.message
        # 注册在组合根上，CLI/MCP/sidecar 从同一清单派生
        assert "valid_ranges" in set(mcp_tools(facade))
        registered = [i for i in tool_inventory(facade) if i.name == "valid_ranges"]
        assert len(registered) == 1 and registered[0].status == "implemented"
