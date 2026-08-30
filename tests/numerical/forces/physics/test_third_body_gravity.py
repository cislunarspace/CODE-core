"""ThirdBodyGravity 物理规律验证。

Rust 单点 ``third_body_acceleration`` / ``indirect_term_acceleration``
与 EphemerisDynamics 第三体分支及解析公式逐点一致（单点求值，毫秒级）。
力分解路径与 EphemerisDynamics 的长弧传播自洽性验证属重度真实计算，
已随端到端测试裁剪移除。

定义与序列化契约见 ``contract/test_third_body_gravity.py``。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.integrators import indirect_term_acceleration, third_body_acceleration

pytestmark = [
    pytest.mark.force,
    pytest.mark.spice,
]

# 公共 SPICE fixtures 来自 tests/conftest.py:
#   spice_manager, spice_eph_system, spice_eph_dynamics, reference_et,
#   reference_epoch


@pytest.fixture
def reference_et(spice_manager, reference_epoch):
    """参考历元的 ET 秒数，与遗留星历动力学契约测试共用。"""
    return spice_manager.utc_to_et(reference_epoch)


@pytest.fixture
def dro_state():
    """月球距离附近的 cislunar 初值（J2000, km, km/s）。

    用于验证第三体间接项的月球距离附近场景。
    """
    return np.array([384400.0, 0.0, 0.0, 0.0, -0.5, 0.0])


@pytest.fixture
def leo_state():
    """近地轨道初始状态（J2000, km, km/s）。"""
    r = 6778.0
    v = np.sqrt(398600.436 / r)
    return np.array([r, 0.0, 0.0, 0.0, v, 0.0])


def _third_body_contribution(dynamics, t, r_sc, body):
    """复现 EphemerisDynamics 单个摄动天体的第三体贡献（直接 + 间接）。

    用同一公式 ``-gm * (r_bsc/|r_bsc|³ + r_ob/|r_ob|³)``，作为
    ``ThirdBodyGravity`` 的独立对照。
    """
    system = dynamics.system
    gm = system.get_gm(body)
    r_ob = np.asarray(system.get_body_position(body, t), dtype=float)
    r_bsc = np.asarray(r_sc, dtype=float) - r_ob
    r_bsc_norm = max(float(np.linalg.norm(r_bsc)), dynamics.MIN_DISTANCE)
    r_ob_norm = max(float(np.linalg.norm(r_ob)), dynamics.MIN_DISTANCE)
    return -gm * (r_bsc / r_bsc_norm**3 + r_ob / r_ob_norm**3)


# =============================================================================
# A. Rust 单点绑定 == EphemerisDynamics 第三体分支
# =============================================================================
class TestThirdBodyAccelMatchesEphemeris:
    """Rust 单点加速度逐字对齐 EphemerisDynamics 的第三体分支。"""

    def test_moon_single_point_matches_ephemeris_branch(
        self, spice_eph_dynamics, reference_et, dro_state
    ):
        """third_body_acceleration("MOON") == EphemerisDynamics 月球第三体增量。"""
        expected = _third_body_contribution(spice_eph_dynamics, reference_et, dro_state[:3], "MOON")

        mu = spice_eph_dynamics.system.get_gm("MOON")
        acc = third_body_acceleration(reference_et, "MOON", "EARTH", dro_state[:3].tolist(), mu)
        np.testing.assert_allclose(acc, expected, atol=1e-12)

    def test_sun_single_point_matches_ephemeris_branch(
        self, spice_eph_dynamics, reference_et, dro_state
    ):
        """third_body_acceleration("SUN") == EphemerisDynamics 太阳第三体增量。"""
        expected = _third_body_contribution(spice_eph_dynamics, reference_et, dro_state[:3], "SUN")

        mu = spice_eph_dynamics.system.get_gm("SUN")
        acc = third_body_acceleration(reference_et, "SUN", "EARTH", dro_state[:3].tolist(), mu)
        np.testing.assert_allclose(acc, expected, atol=1e-12)

    def test_indirect_term_matches_point_mass_formula(self, spice_eph_dynamics, reference_et):
        """indirect_term_acceleration("MOON") == -mu·r_ob/|r_ob|³。"""
        mu = spice_eph_dynamics.system.get_gm("MOON")
        r_ob = np.asarray(spice_eph_dynamics.system.get_body_position("MOON", reference_et))
        expected = -mu / np.linalg.norm(r_ob) ** 3 * r_ob

        acc = indirect_term_acceleration(reference_et, "MOON", "EARTH", mu)
        np.testing.assert_allclose(acc, expected, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
