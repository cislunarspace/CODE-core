"""CR3BP 系统定义、平动点与无量纲尺度的理论测试。"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.dynamics import CR3BP_System, LibrationPoint
from e2m2e.data.constants import SECONDS_PER_DAY, Datum

pytestmark = pytest.mark.theory


@pytest.fixture
def system():
    return CR3BP_System(mu=Datum.DE421.mu, primary="Earth", secondary="Moon")._with_default_scales()


@pytest.mark.parametrize("mu", [0.0, -0.01, 0.5, 1.0])
def test_mass_parameter_must_describe_a_smaller_secondary(mu):
    with pytest.raises(ValueError, match="mu must be in"):
        CR3BP_System(mu=mu, primary="A", secondary="B")


def test_default_earth_moon_scales_match_the_selected_datum(system):
    assert system.characteristic_length == pytest.approx(Datum.DE421.char_length_km)
    assert system.characteristic_time == pytest.approx(Datum.DE421.char_time_s)
    assert system.characteristic_velocity == pytest.approx(
        Datum.DE421.char_length_km / Datum.DE421.char_time_s
    )
    assert pytest.approx(Datum.DE421.char_time_s / SECONDS_PER_DAY) == system.TU


def test_characteristic_scales_follow_their_definitions():
    system = CR3BP_System(mu=Datum.DE421.mu, primary="Earth", secondary="Moon")
    distance = Datum.DE421.char_length_km
    period = 2.0 * np.pi * Datum.DE421.char_time_s

    system.set_characteristic_scales(distance=distance, period=period)

    assert system.characteristic_time == pytest.approx(period / (2.0 * np.pi))
    assert system.characteristic_velocity == pytest.approx(distance / system.characteristic_time)
    assert system.mean_motion == pytest.approx(2.0 * np.pi / period)


@pytest.mark.parametrize("distance, period", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, -1.0)])
def test_characteristic_scales_reject_non_positive_inputs(distance, period):
    system = CR3BP_System(mu=Datum.DE421.mu, primary="Earth", secondary="Moon")
    with pytest.raises(ValueError):
        system.set_characteristic_scales(distance=distance, period=period)


def test_l4_and_l5_follow_the_exact_equilateral_geometry(system):
    points = system.compute_libration_points()
    l4 = points[LibrationPoint.L4]
    l5 = points[LibrationPoint.L5]
    mu = system.mu

    assert_allclose(l4, [0.5 - mu, np.sqrt(3.0) / 2.0, 0.0])
    assert_allclose(l5, [0.5 - mu, -np.sqrt(3.0) / 2.0, 0.0])
    assert_allclose(np.linalg.norm(l4 - np.array([-mu, 0.0, 0.0])), 1.0)
    assert_allclose(np.linalg.norm(l4 - np.array([1.0 - mu, 0.0, 0.0])), 1.0)


def test_collinear_libration_points_satisfy_the_equilibrium_equations(system):
    points = system.compute_libration_points()
    mu = system.mu

    def equilibrium(x):
        r1 = abs(x + mu)
        r2 = abs(x - 1.0 + mu)
        return x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x - 1.0 + mu) / r2**3

    for point in (LibrationPoint.L1, LibrationPoint.L2, LibrationPoint.L3):
        x, y, z = points[point]
        assert y == 0.0
        assert z == 0.0
        assert equilibrium(x) == pytest.approx(0.0, abs=1e-11)


def test_unit_conversion_is_an_inverse_pair(system):
    dimensionless = np.array([0.5, -0.1, 0.2, 0.01, -0.02, 0.03])
    physical = system.dimensionless_to_physical(dimensionless)
    assert_allclose(system.physical_to_dimensionless(physical), dimensionless, atol=1e-12)


def test_unit_conversion_requires_characteristic_scales():
    system = CR3BP_System(mu=Datum.DE421.mu, primary="Earth", secondary="Moon")
    state = np.zeros(6)
    with pytest.raises(ValueError, match="系统未初始化"):
        system.dimensionless_to_physical(state)
    with pytest.raises(ValueError, match="系统未初始化"):
        system.physical_to_dimensionless(state)
