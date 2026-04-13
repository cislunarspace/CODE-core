"""Test that key classes satisfy their declared Protocols."""

import numpy as np
import pytest

from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics, Dynamics
from e2m2e.core.orbit import Orbit
from e2m2e.visualization.config import PlotConfig


class TestCR3BPSystemConformance:
    """CR3BP_System must satisfy the SystemModel protocol."""

    def test_has_mu_attribute(self):
        system = CR3BP_System.from_known_system("earth_moon")
        assert hasattr(system, "mu")
        assert isinstance(system.mu, float)
        assert system.mu > 0

    def test_has_get_jacobi_constant(self):
        system = CR3BP_System.from_known_system("earth_moon")
        assert hasattr(system, "get_jacobi_constant")
        assert callable(system.get_jacobi_constant)

    def test_jacobi_constant_returns_scalar(self):
        system = CR3BP_System.from_known_system("earth_moon")
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
        cj = system.get_jacobi_constant(state)
        assert isinstance(cj, (float, np.floating))


class TestCR3BPDynamicsConformance:
    """CR3BP_Dynamics must satisfy Propagator and EOMProvider protocols."""

    @pytest.fixture()
    def dynamics(self):
        system = CR3BP_System.from_known_system("earth_moon")
        return CR3BP_Dynamics(system)

    def test_satisfies_propagator(self, dynamics):
        assert hasattr(dynamics, "propagate")
        assert callable(dynamics.propagate)

    def test_propagate_returns_correct_shapes(self, dynamics):
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
        result = dynamics.propagate(state, (0, 1.0))
        assert "states" in result
        assert "time" in result
        assert result["states"].shape[1] == 6
        assert result["states"].shape[0] == result["time"].shape[0]

    def test_satisfies_eom_provider(self, dynamics):
        assert hasattr(dynamics, "equations_of_motion")
        assert callable(dynamics.equations_of_motion)

    def test_eom_returns_correct_shape(self, dynamics):
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
        deriv = dynamics.equations_of_motion(0, state)
        assert deriv.shape == (6,)

    def test_eom_output_is_finite(self, dynamics):
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
        deriv = dynamics.equations_of_motion(0, state)
        assert np.all(np.isfinite(deriv))

    def test_inherits_from_dynamics_base(self, dynamics):
        assert isinstance(dynamics, Dynamics)


class TestOrbitConformance:
    """Orbit must satisfy the OrbitContainer protocol."""

    @pytest.fixture()
    def orbit(self):
        system = CR3BP_System.from_known_system("earth_moon")
        states = np.random.randn(100, 6)
        times = np.linspace(0, 1, 100)
        return Orbit(states=states, times=times, system=system)

    def test_has_states(self, orbit):
        assert hasattr(orbit, "states")
        assert orbit.states.shape[1] == 6

    def test_has_times(self, orbit):
        assert hasattr(orbit, "times")
        assert orbit.times.shape[0] == orbit.states.shape[0]

    def test_has_period(self, orbit):
        assert hasattr(orbit, "period")


class TestPlotConfigConformance:
    """PlotConfig must be a Pydantic BaseModel."""

    def test_is_pydantic_model(self):
        from pydantic import BaseModel

        assert issubclass(PlotConfig, BaseModel)

    def test_can_instantiate_with_defaults(self):
        config = PlotConfig()
        assert config is not None


class TestEphemerisDynamicsConformance:
    """EphemerisDynamics must satisfy Propagator and EOMProvider protocols (REQ-025)."""

    @pytest.fixture()
    def eph_dynamics(self):
        from e2m2e.core import SPICEManager, EphemerisSystem, EphemerisDynamics

        system = EphemerisSystem(
            bodies=["EARTH", "MOON", "SUN"],
            spice=SPICEManager(),
        )
        return EphemerisDynamics(system)

    def test_satisfies_propagator(self, eph_dynamics):
        assert hasattr(eph_dynamics, "propagate")
        assert callable(eph_dynamics.propagate)

    def test_satisfies_eom_provider(self, eph_dynamics):
        assert hasattr(eph_dynamics, "equations_of_motion")
        assert callable(eph_dynamics.equations_of_motion)

    def test_inherits_from_dynamics_base(self, eph_dynamics):
        assert isinstance(eph_dynamics, Dynamics)

    def test_runtime_checkable_propagator(self, eph_dynamics):
        from e2m2e.mbse.architecture.ports import Propagator

        assert isinstance(eph_dynamics, Propagator)

    def test_runtime_checkable_eom_provider(self, eph_dynamics):
        from e2m2e.mbse.architecture.ports import EOMProvider

        assert isinstance(eph_dynamics, EOMProvider)
