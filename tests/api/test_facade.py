"""Facade 公共行为、错误翻译与工具元数据测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from api.conftest import control_orbit_business_parameters
from e2m2e.api.config import Config
from e2m2e.api.facade import Facade, mcp_tools, tool_inventory
from e2m2e.api.models import (
    ControlOrbitRequest,
    DesignOrbitRequest,
    FamilyGenerationRequest,
    OrbitError,
    PropagationRequest,
    SpacetimeTransformRequest,
    TransferDesignRequest,
)
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.interface


def _fake_control_result():
    return SimpleNamespace(
        num_failed=0,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="任务完成",
        sk_statistic=SimpleNamespace(rows=np.zeros((2, 3)), num_failed=0),
        maneuvers=SimpleNamespace(mjd_tdb=np.array([60000.0]), delta_v_mps=np.array([1.0])),
        controlled_ephemeris=None,
    )


class TestFacadeValidation:
    @pytest.mark.parametrize(
        "call",
        [
            lambda facade: facade.design_orbit(orbit_type="DRO", duration=0.0),
            lambda facade: facade.control_orbit(control_mode=9),
            lambda facade: facade.transfer_design(transfer_type="HMN"),
            lambda facade: facade.orbit_propagation(),
            lambda facade: facade.spacetime_transform(),
            lambda facade: facade.orbit_family_generation(),
        ],
    )
    def test_invalid_input_is_translated_to_orbit_error(self, call):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            call(Facade())


class TestFacadeDelegation:
    def test_control_upper_bounds_are_translated_to_orbit_error(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.control_orbit(input_ephemeris="x", num_controls=10001)
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.control_orbit(input_ephemeris="x", num_monte_carlo=1001)

    def test_design_preserves_algorithm_status(self, monkeypatch):
        from e2m2e.algorithm.design import DesignNotConvergedError

        def fail_design(*args, **kwargs):
            raise DesignNotConvergedError(
                "修正未收敛",
                status=ConvergenceState.MAX_ITERATIONS,
                cause=FailureCause.MAX_ITERATIONS_REACHED,
            )

        monkeypatch.setattr("e2m2e.algorithm.design.design_orbit", fail_design)
        with pytest.raises(OrbitError) as exc_info:
            Facade().design_orbit(orbit_type="DRO")
        assert exc_info.value.status is ConvergenceState.MAX_ITERATIONS
        assert exc_info.value.cause is FailureCause.MAX_ITERATIONS_REACHED

    def test_control_passes_model_values_and_config_to_algorithm(self, monkeypatch):
        import e2m2e.algorithm.station_keeping as station_keeping

        captured: dict[str, Any] = {}
        business = set(control_orbit_business_parameters())

        def fake_control(input_ephemeris, **kwargs):
            captured.update(kwargs)
            return _fake_control_result()

        monkeypatch.setattr(station_keeping, "control_orbit", fake_control)
        facade = Facade(config=Config(kernel_dir="configured-kernels"))
        facade.control_orbit(
            input_ephemeris="x",
            control_interval=45.0,
            feedback_arc=20.0,
            position_accuracy=123.0,
            thrust_mean=8.0,
            tight_tolerance_km=0.5,
        )

        assert set(captured) == business | {"kernel_dir"}
        assert captured["kernel_dir"] == "configured-kernels"
        assert captured["control_interval"] == 45.0
        assert captured["feedback_arc"] == 20.0
        assert captured["position_accuracy"] == 123.0
        assert captured["thrust_mean"] == 8.0
        assert captured["tight_tolerance_km"] == 0.5
        assert captured["num_controls"] == 120


class TestFacadeCallChains:
    def test_unknown_design_type_is_translated_to_orbit_error(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().design_orbit(orbit_type="NOPE")

    def test_hmn_returns_converged_response(self):
        response = Facade().transfer_design(
            transfer_type="HMN",
            tli_epoch="2025-06-21T11:00:00",
            target_orbit_radius_km=42164.0,
        )
        assert response.status is ConvergenceState.CONVERGED
        assert response.cause is FailureCause.NONE
        assert response.message == "霍曼转移完成"

    def test_unknown_transfer_type_is_not_implemented_error(self):
        with pytest.raises(OrbitError, match="NOT_IMPLEMENTED"):
            Facade().transfer_design(
                transfer_type="UNKNOWN_TYPE",
                tli_epoch="2025-06-21T11:00:00",
                target_orbit_radius_km=42164.0,
            )

    def test_unknown_spacetime_transform_is_translated_to_orbit_error(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().spacetime_transform(
                states=[[1.0] * 6],
                times=[0.0],
                transform_type="unknown",
                et0_jd=2459000.0,
            )

    def test_spacetime_transform_rejects_mismatched_batches(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().spacetime_transform(
                states=[[1.0] * 6, [2.0] * 6],
                times=[0.0],
                transform_type="j2000_to_synodic",
                et0_jd=2459000.0,
            )

    def test_halo_family_delegates_to_algorithm(self):
        family = Facade().orbit_family_generation(
            orbit_type="HALO", libration_point=1, max_amplitude_km=3000.0, n_orbits=2
        )
        assert family.family_type == "halo"
        assert len(family) >= 1
        assert all(orbit.parameters.get("libration_point") == 1 for orbit in family)

    def test_non_halo_family_reports_not_implemented(self):
        with pytest.raises(OrbitError, match="NOT_IMPLEMENTED"):
            Facade().orbit_family_generation(orbit_type="SPO", libration_point=4)

    @pytest.mark.parametrize(
        "params",
        [{"orbit_type": "NOPE"}, {"orbit_type": "HALO", "libration_point": 4}],
    )
    def test_invalid_family_input_is_translated_to_orbit_error(self, params):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().orbit_family_generation(**params)

    def test_invalid_family_libration_point_type_is_translated_to_orbit_error(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().orbit_family_generation(orbit_type="HALO", libration_point="bad")


class TestFacadeToolInventory:
    def test_inventory_is_derived_from_exposed_methods(self):
        facade = Facade()
        names = set(mcp_tools(facade))
        inventory = tool_inventory(facade)
        assert names == {tool.name for tool in inventory}
        assert len(inventory) == 12
        assert all(tool.mcp_exposed for tool in inventory)

    def test_inventory_distinguishes_implemented_and_placeholder_tools(self):
        by_name = {tool.name: tool for tool in tool_inventory(Facade())}
        implemented = {
            "design_orbit": DesignOrbitRequest,
            "control_orbit": ControlOrbitRequest,
            "transfer_design": TransferDesignRequest,
            "orbit_propagation": PropagationRequest,
            "spacetime_transform": SpacetimeTransformRequest,
            "orbit_family_generation": FamilyGenerationRequest,
        }
        placeholders = {
            "transfer_search",
            "low_thrust_design",
            "manifold_analysis",
            "low_energy_transfer",
            "relative_motion",
        }
        assert all(by_name[name].status == "implemented" for name in implemented)
        assert all(by_name[name].request_model is model for name, model in implemented.items())
        assert all(by_name[name].status == "placeholder" for name in placeholders)
        assert all(by_name[name].request_model is None for name in placeholders)
        assert by_name["orbit_stability"].status == "implemented"
        assert by_name["orbit_stability"].request_model is None
