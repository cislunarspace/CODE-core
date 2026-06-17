"""算法测试共享 fixtures。

为 DRO 种子轨道修正场景提供预配置 fixture。
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from e2m2e.algorithms import Continuation, DifferentialCorrection
from e2m2e.core import CR3BP_Dynamics, Orbit

# DRO seed parameters (Cui et al. 2025) — standardise the suite on this seed.
DRO_X0 = 0.79188556619742
DRO_VY0 = 0.573665890385585
DRO_PERIOD_GUESS = 6.307498


@pytest.fixture(scope="session")
def dro_seed_state() -> np.ndarray:
    """DRO seed state vector [x, y, z, vx, vy, vz] in dimensionless CR3BP units."""
    return np.array([DRO_X0, 0.0, 0.0, 0.0, DRO_VY0, 0.0])


@pytest.fixture(scope="session")
def dro_seed_orbit(dro_seed_state) -> Orbit:
    """DRO seed Orbit object (1-point orbit) ready for differential correction."""
    orbit = Orbit(states=[dro_seed_state], times=[0])
    orbit.period = DRO_PERIOD_GUESS
    return orbit


@pytest.fixture(scope="session")
def _corrected_dro_cached(dro_seed_orbit) -> Orbit:
    """Compute corrected DRO once per session; expensive (5–15 STM propagations).

    The corrector is built here (not via dro_corrector) so the cached result
    doesn't depend on a function-scoped fixture.
    """
    system = _make_earth_moon_system()
    dynamics = CR3BP_Dynamics(system)
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(DRO_X0)
    return corrector.iterate_correction(dro_seed_orbit, verbose=False)


@pytest.fixture
def corrected_dro(_corrected_dro_cached) -> Orbit:
    """Fresh deepcopy of the corrected DRO for each test (safe for mutation)."""
    return copy.deepcopy(_corrected_dro_cached)


@pytest.fixture
def dro_corrector() -> DifferentialCorrection:
    """Fresh DifferentialCorrection configured for the standard DRO correction."""
    system = _make_earth_moon_system()
    dynamics = CR3BP_Dynamics(system)
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(DRO_X0)
    return corrector


@pytest.fixture
def dro_dynamics() -> CR3BP_Dynamics:
    """Earth-Moon CR3BP dynamics for DRO algorithm tests."""
    system = _make_earth_moon_system()
    return CR3BP_Dynamics(system)


@pytest.fixture
def dro_continuation(dro_corrector) -> Continuation:
    """Continuation instance configured for DRO family generation."""
    return Continuation(corrector=dro_corrector, step=0.001)


def _make_earth_moon_system():
    """Build an Earth-Moon CR3BP system matching the existing test suite's construction.

    The root conftest's `earth_moon_system` is function-scoped (so tests can
    mutate it freely). Session-scoped fixtures here need a fresh system they
    own; this helper is the single place that decision lives.
    """
    from e2m2e.core.cr3bp_system import CR3BP_System

    return CR3BP_System(mu=1.21506683e-2, primary="earth", secondary="moon")
