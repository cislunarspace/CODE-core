"""Orbit 状态按时间重传播的积分器行为测试。"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.integrator


def _periodic_orbit():
    from e2m2e.data.types.orbit import Orbit

    orbit = Orbit(
        states=np.array([[0.8, 0.0, 0.0, 0.0, 0.1, 0.0]]),
        times=np.array([0.0]),
    )
    orbit.period = 1.0
    return orbit


def test_orbit_state_at_initial_epoch_matches_the_input_state(earth_moon_dynamics):
    orbit = _periodic_orbit()

    state = earth_moon_dynamics.propagate_orbit_state_at_time(orbit, 0.0)

    assert_allclose(state, orbit.states[0], rtol=1e-9, atol=1e-12)


def test_orbit_state_at_time_returns_a_finite_state(earth_moon_dynamics):
    state = earth_moon_dynamics.propagate_orbit_state_at_time(
        _periodic_orbit(), 0.05, integration_dt=0.005
    )

    assert state.shape == (6,)
    assert np.all(np.isfinite(state))
