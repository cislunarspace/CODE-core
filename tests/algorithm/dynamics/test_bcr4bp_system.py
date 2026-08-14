"""BCR4BP 太阳解析轨道与无量纲参数的理论测试。"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.dynamics import BCR4BPSystem
from e2m2e.data.constants import SECONDS_PER_DAY, Datum

pytestmark = pytest.mark.theory


@pytest.fixture
def system():
    return BCR4BPSystem.earth_moon()


def test_solar_parameters_are_derived_from_the_selected_constants(system):
    assert system.sun_mass == pytest.approx(Datum.DE440.sun_gm / Datum.DE440.emb_gm)
    assert system.gravitational_parameter("sun") == pytest.approx(system.sun_mass)
    assert system.gravitational_parameter("primary") == pytest.approx(1.0 - system.mu)
    assert system.gravitational_parameter("secondary") == pytest.approx(system.mu)


def test_solar_synodic_rate_is_derived_from_the_characteristic_period(system):
    expected = system.orbital_period / system.YEAR - 1.0
    assert system.sun_angular_rate == pytest.approx(expected)
    assert system.sun_angular_rate < 0.0


def test_solar_position_follows_the_analytic_coplanar_circle(system):
    for time in (0.0, 0.7, 3.3):
        position = system.sun_position(time)
        assert np.linalg.norm(position) == pytest.approx(system.sun_distance, rel=1e-14)
        assert position[2] == 0.0

    time = 1.234
    angle = system.sun_angular_rate * time
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    assert_allclose(system.sun_position(time), rotation @ system.sun_position(0.0), atol=1e-12)


def test_initial_solar_phase_controls_the_initial_direction():
    system = BCR4BPSystem.earth_moon(sun_phase0=np.pi / 2.0)
    position = system.sun_position(0.0)
    assert position[0] == pytest.approx(0.0, abs=1e-10)
    assert position[1] == pytest.approx(system.sun_distance)


def test_solar_synodic_period_has_the_expected_units(system):
    period_days = (
        2.0 * np.pi / abs(system.sun_angular_rate) * system.characteristic_time / SECONDS_PER_DAY
    )
    assert period_days == pytest.approx(29.53, abs=0.1)
