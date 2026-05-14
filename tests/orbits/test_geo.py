"""
测试 e2m2e.orbits.geo 模块

从 transfer-orbit-design/tests/test_inbound_transfer.py::TestGeoUtils 移植。
"""

import numpy as np

from e2m2e.orbits.geo import (
    DU,
    EARTH_CENTER,
    MU,
    R_GEO,
    T_GEO,
    V_CIRCULAR_GEO,
    VU,
    check_collision,
    compute_departure_velocity,
    detect_geo_sphere_crossing,
    geo_circular_velocity_rotating,
)


class TestGeoConstants:
    def test_r_geo_km(self):
        assert abs(R_GEO * DU - 42164.0) < 1.0

    def test_v_circular_geo(self):
        assert abs(V_CIRCULAR_GEO * VU - 3071.0) < 5.0

    def test_t_geo_period(self):
        assert abs(T_GEO * 4.34811305 - 1.0) < 0.01

    def test_earth_center(self):
        np.testing.assert_allclose(EARTH_CENTER, [-MU, 0.0, 0.0])


class TestGeoCircularVelocity:
    def test_positive_y_velocity(self):
        pos = EARTH_CENTER + np.array([0.11, 0.0, 0.0])
        vel = geo_circular_velocity_rotating(pos)
        assert abs(vel[1]) > 0
        assert vel[2] == 0.0


class TestComputeDepartureVelocity:
    def test_alpha_one_unchanged(self):
        state = np.array([0.1, 0.0, 0.0, 0.0, 1.0, 0.0])
        v_new = compute_departure_velocity(state, 1.0)
        np.testing.assert_allclose(v_new, state[3:], atol=1e-12)

    def test_alpha_two_doubles_tangential(self):
        state = np.array([0.1, 0.0, 0.0, 0.0, 1.0, 0.0])
        v_new2 = compute_departure_velocity(state, 2.0)
        np.testing.assert_allclose(v_new2, [0.0, 2.0, 0.0], atol=1e-12)


class TestCheckCollision:
    def test_earth_collision(self):
        states = np.array([[EARTH_CENTER[0] + 100 / DU, 0, 0, 0, 0, 0]])
        found, body, idx = check_collision(states, 0.01215, 200 / DU, 100 / DU)
        assert found
        assert body == "earth"


class TestDetectGeoSphereCrossing:
    def test_crossing_detected(self):
        n = 100
        states = np.zeros((n, 6))
        for i in range(n):
            t = i / (n - 1)
            r = R_GEO * (0.9 + 0.2 * t)
            states[i, :3] = EARTH_CENTER + np.array([r, 0, 0])

        crossed, idx, dist = detect_geo_sphere_crossing(states)
        assert crossed
        assert idx >= 0
