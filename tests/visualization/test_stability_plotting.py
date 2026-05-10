"""Integration tests: StabilityAnalysis → FamilyPlotter.

Verifies that stability values computed by algorithms.stability.StabilityAnalysis
can be passed directly to FamilyPlotter plotting methods without errors.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from e2m2e.algorithms.stability import StabilityAnalysis
from e2m2e.core import CR3BP_Dynamics, CR3BP_System, Orbit, OrbitFamily
from e2m2e.visualization import FamilyPlotter


MU = 1.21506683e-2


def _make_orbit(x0: float, period: float, system: CR3BP_System) -> Orbit:
    t = np.linspace(0, period, 50)
    states = np.zeros((len(t), 6))
    states[:, 0] = x0
    states[:, 1] = 0.05 * np.sin(2 * np.pi * t / period)
    states[:, 4] = 0.1 * np.cos(2 * np.pi * t / period)
    orbit = Orbit(states=states, times=t)
    orbit.period = period
    orbit.system = system
    return orbit


def _compute_stability_values(
    family: OrbitFamily, dynamics: CR3BP_Dynamics
) -> list[float]:
    values = []
    for orbit in family:
        if orbit.period is None:
            values.append(1.0)
            continue
        sa = StabilityAnalysis(orbit, dynamics)
        sa.compute_floquet_multipliers()
        classification = sa.classify_orbit()
        values.append(float(classification["max_eigenvalue_magnitude"]))
    return values


@pytest.fixture
def system():
    sys = CR3BP_System(mu=MU, primary="earth", secondary="moon")
    sys.compute_libration_points()
    return sys


@pytest.fixture
def dynamics(system):
    return CR3BP_Dynamics(system)


@pytest.fixture
def family(system):
    fam = OrbitFamily(family_type="test_dro")
    for i in range(3):
        fam.add_orbit(_make_orbit(0.79 + i * 0.005, 3.0 + i * 0.2, system))
    return fam


class TestStabilityToFamilyPlotter:
    """StabilityAnalysis output feeds into FamilyPlotter without errors."""

    def test_plot_jacobi_period_stability_accepts_stability_values(
        self, system, dynamics, family
    ):
        stability_values = _compute_stability_values(family, dynamics)
        jacobi_values = [3.17 - i * 0.001 for i in range(len(family))]
        periods = [o.period for o in family]

        plotter = FamilyPlotter(system)
        fig, ax = plotter.plot_jacobi_period_stability(
            jacobi_values=jacobi_values,
            periods=periods,
            stability_values=stability_values,
            show=False,
        )

        assert fig is not None
        assert len(stability_values) == len(family)

    def test_plot_family_overview_accepts_stability_values(
        self, system, dynamics, family
    ):
        stability_values = _compute_stability_values(family, dynamics)
        jacobi_values = [3.17 - i * 0.001 for i in range(len(family))]
        periods = [o.period for o in family]

        plotter = FamilyPlotter(system)
        fig = plotter.plot_family_overview(
            family_result=family,
            jacobi_values=jacobi_values,
            periods=periods,
            stability_values=stability_values,
            show=False,
        )

        assert fig is not None
        assert len(stability_values) == len(family)

    def test_orbit_without_period_yields_default_stability(
        self, system, dynamics
    ):
        fam = OrbitFamily(family_type="no_period")
        orbit = _make_orbit(0.8, 3.0, system)
        orbit.period = None
        fam.add_orbit(orbit)

        values = _compute_stability_values(fam, dynamics)

        assert len(values) == 1
        assert values[0] == 1.0

    def test_empty_family_yields_empty_stability(self, system, dynamics):
        fam = OrbitFamily(family_type="empty")

        values = _compute_stability_values(fam, dynamics)

        assert values == []

    def test_stability_values_are_nonnegative_floats(
        self, system, dynamics, family
    ):
        values = _compute_stability_values(family, dynamics)

        for v in values:
            assert isinstance(v, float)
            assert v >= 0
