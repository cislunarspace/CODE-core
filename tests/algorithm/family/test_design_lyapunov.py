"""Lyapunov CR3BP 周期轨道设计契约。"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.family.cr3bp_orbits import design_lyapunov
from e2m2e.integrators import orbit_family_metric_py

pytestmark = pytest.mark.orchestration


def test_design_lyapunov_hits_amplitude_and_closes_planar_orbit() -> None:
    target_km = 12000.0
    orbit = design_lyapunov(1, target_km)

    assert orbit.period is not None
    assert orbit.closure_error is not None
    assert orbit.closure_error < 1e-6
    assert orbit.states[0, 2] == 0.0
    assert orbit.states[0, 5] == 0.0

    system = orbit.system
    assert system is not None
    du = system.characteristic_length
    assert du is not None
    _, max_y = orbit_family_metric_py(
        float(system.mu),
        "y-amplitude",
        0,
        orbit.states[0],
        float(orbit.period),
        sample_count=1000,
    )
    assert abs(float(max_y) * du - target_km) <= 20.0
