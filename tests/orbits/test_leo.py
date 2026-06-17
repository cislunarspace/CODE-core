"""e2m2e.orbits.leo 模块测试。

覆盖 LEO 常数、圆速度、轨道状态生成。
"""

import numpy as np

from e2m2e.orbits.leo import (
    DU,
    EARTH_CENTER,
    R_LEO,
    T_LEO,
    V_CIRCULAR_LEO,
    generate_leo_orbit_states,
    leo_circular_velocity_rotating,
)


class TestLeoConstants:
    def test_r_leo_km(self):
        assert abs(R_LEO * DU - 6771.0) < 1.0

    def test_v_circular_leo(self):
        # VU not re-exported from leo, compute manually
        from e2m2e.orbits.geo import VU

        v_ms = V_CIRCULAR_LEO * VU
        assert 7500 < v_ms < 8000

    def test_t_leo_period(self):
        t_days = T_LEO * 4.34811305
        assert 0.05 < t_days < 0.07


class TestLeoCircularVelocity:
    def test_positive_y_velocity(self):
        pos = EARTH_CENTER + np.array([R_LEO, 0, 0])
        vel = leo_circular_velocity_rotating(pos)
        assert abs(vel[1]) > 0
        assert vel[2] == 0.0


class TestGenerateLeoOrbitStates:
    def test_shape(self):
        states = generate_leo_orbit_states(100)
        assert states.shape == (100, 6)

    def test_distances_match_r_leo(self):
        states = generate_leo_orbit_states(100)
        dists = np.linalg.norm(states[:, :3] - EARTH_CENTER, axis=1)
        np.testing.assert_allclose(dists, R_LEO, rtol=1e-10)

    def test_planar(self):
        states = generate_leo_orbit_states(100)
        np.testing.assert_allclose(states[:, 2], 0.0, atol=1e-15)
