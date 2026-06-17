"""e2m2e.orbits.geo 未覆盖分支测试。

覆盖原点圆速度回退、无穿越检测与最近接近搜索。
"""

import numpy as np
import pytest

from e2m2e.orbits.geo import (
    EARTH_CENTER,
    MU,
    R_GEO,
    check_collision,
    compute_departure_velocity,
    compute_geo_dv2,
    detect_geo_sphere_crossing,
    find_closest_approach_to_geo,
    geo_circular_velocity_rotating,
)


class TestGeoCircularVelocityAtOrigin:
    def test_at_earth_center_returns_fallback(self):
        vel = geo_circular_velocity_rotating(EARTH_CENTER)
        assert vel[2] == 0.0
        assert abs(vel[1]) > 0 or abs(vel[0]) > 0


class TestDetectGeoSphereNoCrossing:
    def test_all_inside_no_crossing(self):
        n = 50
        states = np.zeros((n, 6))
        for i in range(n):
            r = R_GEO * 0.5
            states[i, :3] = EARTH_CENTER + np.array([r, 0, 0])
        crossed, idx, dist = detect_geo_sphere_crossing(states)
        assert not crossed
        assert idx == -1
        assert dist == 0.0


class TestFindClosestApproachToGeo:
    def test_finds_minimum(self):
        n = 50
        states = np.zeros((n, 6))
        for i in range(n):
            r = R_GEO * (0.8 + 0.4 * i / (n - 1))
            states[i, :3] = EARTH_CENTER + np.array([r, 0, 0])
        sphere_dist, idx = find_closest_approach_to_geo(states)
        assert sphere_dist >= 0
        assert 0 <= idx < n


class TestComputeGeoDv2:
    def test_nonzero_dv_at_arbitrary_state(self):
        pos = EARTH_CENTER + np.array([R_GEO, 0, 0])
        state = np.array([*pos, 0.0, 0.0, 0.0])
        dv = compute_geo_dv2(state)
        assert dv > 0


class TestComputeDepartureVelocityAtOrigin:
    def test_near_origin_returns_copy(self):
        state = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        result = compute_departure_velocity(state, 2.0)
        np.testing.assert_allclose(result, state[3:])


class TestCheckCollisionEdgeCases:
    def test_negative_radius_raises(self):
        states = np.zeros((5, 6))
        with pytest.raises(ValueError, match="Radii must be positive"):
            check_collision(states, MU, -1.0, 1.0)

    def test_moon_collision(self):
        moon_center = np.array([1 - MU, 0, 0])
        states = np.zeros((1, 6))
        states[0, :3] = moon_center + np.array([0.0001, 0, 0])
        found, body, idx = check_collision(states, MU, 0.1, 0.01)
        assert found
        assert body == "moon"

    def test_no_collision(self):
        states = np.zeros((1, 6))
        states[0, :3] = np.array([0.5, 0.5, 0.0])
        found, body, idx = check_collision(states, MU, 0.05, 0.05)
        assert not found
        assert body is None
        assert idx == -1
