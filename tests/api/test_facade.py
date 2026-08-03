"""api 层 Facade 门面测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from e2m2e.api.facade import Facade, mcp_tools
from e2m2e.api.models import ControlOrbitRequest, DesignOrbitRequest, OrbitError


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

    def test_placeholder_methods(self):
        facade = Facade()
        for name in ("transfer_design", "orbit_propagation", "spacetime_transform"):
            with pytest.raises(NotImplementedError):
                getattr(facade, name)()

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
        assert "orbit_family_generation" in names
