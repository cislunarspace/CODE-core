"""spatiography 三工具的接口测试（Facade 层契约与异常翻译）。"""

from __future__ import annotations

import pytest

from e2m2e.api.facade import Facade
from e2m2e.api.models import (
    OrbitError,
    SpatiographyBoundariesRequest,
    SpatiographyBoundariesResponse,
    SpatiographyClassifyRequest,
    SpatiographyScalesRequest,
)

pytestmark = pytest.mark.interface


class TestSpatiographyScales:
    def test_golden_values_travel_through_facade(self):
        response = Facade().spatiography_scales()
        assert response.status.value == "converged"
        assert response.scales["laplace_radius_geolunar_km"] == pytest.approx(48812.40, rel=1e-4)
        assert response.scales["hill_radius_moon_km"] == pytest.approx(61364.0, rel=1e-4)
        assert len(response.resonance_ladder) == 40
        assert response.libration_points_km["L1"][1] == 0.0
        assert "Rosengren" in response.citation

    def test_element_filter_and_unknown_name(self):
        facade = Facade()
        filtered = facade.spatiography_scales(elements=["hill_radius_moon_km"])
        assert set(filtered.scales) == {"hill_radius_moon_km"}
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.spatiography_scales(elements=["bogus_scale"])

    def test_over_a_moon_derived_keys(self):
        scales = Facade().spatiography_scales().scales
        assert scales["laplace_radius_geolunar_over_a_moon"] == pytest.approx(0.12732, rel=1e-3)


class TestSpatiographyClassify:
    def test_classify_states_with_legend(self):
        response = Facade().spatiography_classify(
            states=[[0.5, 0.1, 0.0, 0.0, 1.0, 0.0]],
            frame="synodic_barycentric_nd",
        )
        assert response.zone_ids == [[2]]
        assert response.legend["2"] == "cislunar_outer_resonant"
        assert response.legend["3"] == "circumlunar"
        diag = response.diagnostics[0]
        assert diag["topology_case"] == 3

    def test_invalid_state_shape_translated(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().spatiography_classify(states=[[1.0, 2.0, 3.0]], frame="synodic_barycentric_nd")

    def test_frame_description_locks_state_frame_vocabulary(self):
        """frame 字段 description 须声明 ADR 0040 state_frame 词汇与双口径。"""
        desc = SpatiographyClassifyRequest.model_json_schema()["properties"]["frame"]["description"]
        assert "synodic_barycentric_km" in desc
        assert "synodic_barycentric_nd" in desc
        assert "质心原点" in desc


class TestSpatiographyBoundaries:
    def test_synodic_planar_payload_structure(self):
        response = Facade().spatiography_boundaries(resolution=32)
        assert response.state_frame == "synodic_barycentric_km"
        assert len(response.elements) == 13
        circle = next(e for e in response.elements if e["kind"] == "circle")
        assert circle["radius_km"] == pytest.approx(48812.40, rel=1e-4)
        assert len(circle["points_km"]) == 32

    def test_ae_curves_payload_uses_element_space_label(self):
        response = Facade().spatiography_boundaries(
            kind="ae_curves", boundary_set=["resonance_verticals"]
        )
        assert response.state_frame == "element_space_ae"
        assert all(e["kind"] == "vertical_ae" for e in response.elements)

    def test_unknown_boundary_set_translated(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().spatiography_boundaries(boundary_set=["bogus"])

    def test_state_frame_description_documents_new_vocabulary(self):
        desc = SpatiographyBoundariesResponse.model_json_schema()["properties"]["state_frame"][
            "description"
        ]
        assert "element_space_ae" in desc
        assert "synodic_barycentric_km" in desc

    def test_request_schema_is_self_describing(self):
        schema = SpatiographyScalesRequest.model_json_schema()
        assert schema["properties"]["system"]["default"] == "earth_moon"
        assert (
            SpatiographyBoundariesRequest.model_json_schema()["properties"]["resolution"]["default"]
            == 720
        )
