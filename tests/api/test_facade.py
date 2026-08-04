"""api 层 Facade 门面测试。"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from e2m2e.api.facade import Facade, mcp_tools
from e2m2e.api.models import (
    ControlOrbitRequest,
    DesignOrbitRequest,
    OrbitError,
    PropagationRequest,
    SpacetimeTransformRequest,
    TransferDesignRequest,
)


class TestDesignOrbitRequest:
    def test_defaults(self):
        req = DesignOrbitRequest(orbit_type="DRO")
        assert req.duration == 1.0
        assert req.output_step == 3600.0
        assert req.correction_method == "two_level"

    def test_duration_bounds(self):
        with pytest.raises(ValidationError):
            DesignOrbitRequest(orbit_type="DRO", duration=0.0)
        with pytest.raises(ValidationError):
            DesignOrbitRequest(orbit_type="DRO", duration=21.0)

    def test_perilune_height_bounds(self):
        with pytest.raises(ValidationError):
            DesignOrbitRequest(orbit_type="NRHO", perilune_height=50.0)


class TestControlOrbitRequest:
    def test_control_mode_bounds(self):
        """mode 7 超出范围；mode 4 在 API 层允许（算法层校验 engine_layout）。"""
        with pytest.raises(ValidationError):
            ControlOrbitRequest(input_ephemeris="x", control_mode=7)
        # mode 4 在 API 层不报错（engine_layout 校验在算法层）
        req = ControlOrbitRequest(input_ephemeris="x", control_mode=4)
        assert req.control_mode == 4


class TestTransferDesignRequest:
    def test_defaults(self):
        req = TransferDesignRequest(transfer_type="HMN", tli_epoch="2025-06-21T11:00:00")
        assert req.parking_alt_km == 200.0
        assert req.incl_deg == 28.5
        assert req.flight_path_deg == 0.0

    def test_invalid_transfer_type_type(self):
        with pytest.raises(ValidationError):
            TransferDesignRequest(transfer_type=123, tli_epoch="2025-06-21T11:00:00")


class TestPropagationRequest:
    def test_defaults(self):
        req = PropagationRequest(
            initial_state=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            epoch="2025-06-21T11:00:00",
            duration=3600.0,
        )
        assert req.output_step == 3600.0
        assert req.force_config is None

    def test_duration_must_be_positive(self):
        with pytest.raises(ValidationError):
            PropagationRequest(
                initial_state=[0.0] * 6,
                epoch="2025-06-21T11:00:00",
                duration=0.0,
            )

    def test_invalid_state_shape(self):
        # Pydantic 在 API 边界强制长度为 6
        with pytest.raises(ValidationError):
            PropagationRequest(
                initial_state=[0.0] * 5,
                epoch="2025-06-21T11:00:00",
                duration=3600.0,
            )


class TestSpacetimeTransformRequest:
    def test_defaults(self):
        req = SpacetimeTransformRequest(
            states=[[1.0] * 6],
            times=[0.0],
            transform_type="j2000_to_synodic",
            et0_jd=2459000.0,
        )
        assert req.ephemeris_path is None

    def test_unknown_transform_type_type(self):
        with pytest.raises(ValidationError):
            SpacetimeTransformRequest(
                states=[[1.0] * 6],
                times=[0.0],
                transform_type=123,
                et0_jd=2459000.0,
            )


class TestFacade:
    def test_construct(self):
        facade = Facade()
        assert facade._config is not None

    def test_design_orbit_invalid_params(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.design_orbit(orbit_type="DRO", duration=0.0)

    def test_design_orbit_unknown_type(self):
        facade = Facade()
        with pytest.raises(OrbitError):
            facade.design_orbit(orbit_type="NOPE")

    def test_control_orbit_invalid_params(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.control_orbit(control_mode=9)

    def test_transfer_design_invalid_params(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.transfer_design(transfer_type="HMN")

    def test_orbit_propagation_invalid_params(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.orbit_propagation(
                initial_state=[0.0] * 6,
                epoch="2025-06-21T11:00:00",
                duration=0.0,
            )

    def test_spacetime_transform_invalid_params(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.spacetime_transform(
                states=[[1.0] * 6],
                times=[0.0],
                transform_type="unknown",
                et0_jd=2459000.0,
            )

    def test_orbit_family_generation_unknown(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.orbit_family_generation(orbit_type="NOPE")


class TestMcpTools:
    def test_derives_exposed_methods(self):
        facade = Facade()
        names = mcp_tools(facade)
        assert "design_orbit" in names
        assert "control_orbit" in names
        assert "transfer_design" in names
        assert "orbit_propagation" in names
        assert "spacetime_transform" in names
        assert "orbit_family_generation" in names


class TestFacadeCallChain:
    """无需 SPICE 的轻量调用链：仅验证错误码与序列化路径。"""

    def test_transfer_design_hmn_call_chain(self):
        facade = Facade()
        response = facade.transfer_design(
            transfer_type="HMN",
            tli_epoch="2025-06-21T11:00:00",
            target_orbit_radius_km=42164.0,
        )
        assert response.transfer_type == "HMN"
        assert response.delta_v > 0.0
        assert "departure_state" in response.details or "delta_v_theory" in response.details

    def test_transfer_design_invalid_type_call_chain(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="NOT_IMPLEMENTED"):
            facade.transfer_design(
                transfer_type="UNKNOWN_TYPE",
                tli_epoch="2025-06-21T11:00:00",
                target_orbit_radius_km=42164.0,
            )

    def test_spacetime_transform_mismatched_lengths(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.spacetime_transform(
                states=[[1.0] * 6, [2.0] * 6],
                times=[0.0],
                transform_type="j2000_to_synodic",
                et0_jd=2459000.0,
            )

    def test_details_to_dict_with_ndarray(self):
        from e2m2e.api.facade import _details_to_dict

        class DummyDataclass:
            pass

        details = {
            "array": np.array([1.0, 2.0]),
            "nested": {"tuple": (np.array([3.0]), 4.0)},
        }
        result = _details_to_dict(details)
        assert result["array"] == [1.0, 2.0]
        assert result["nested"]["tuple"] == [[3.0], 4.0]
