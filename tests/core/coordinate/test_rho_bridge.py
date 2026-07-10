"""rho 无量纲坐标 ↔ ECI（J2000, km）桥接测试。

覆盖：往返一致性、物理合理性、旋转矩阵正交性。
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithms.normal_form.context import NormalFormContext
from e2m2e.core import LibrationPoint

pytestmark = pytest.mark.spice


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def context(earth_moon_system):
    """L1 平动点的 NormalFormContext。"""
    return NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=2451545.0,  # J2000.0
        order=4,
    )


@pytest.fixture
def context_l2(earth_moon_system):
    """L2 平动点的 NormalFormContext。"""
    return NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L2,
        epoch=2451545.0,
        order=4,
    )


@pytest.fixture
def system(spice_manager):
    """地球-月球 EphemerisSystem。"""
    from e2m2e.core.ephemeris_system import EphemerisSystem

    return EphemerisSystem(
        bodies=["MOON", "SUN"],
        spice=spice_manager,
        origin="EARTH",
    )


# =============================================================================
# Test compute_emr_rotation
# =============================================================================


class TestEMRRotation:
    """测试 EMR 旋转矩阵。"""

    def test_rotation_is_orthogonal(self, system):
        """C(t) 应正交：det=1, C@C.T=I。"""
        from e2m2e.core.rho_bridge import compute_emr_rotation

        et = system.spice.utc_to_et("2025-06-21T11:00:06")
        C, _ = compute_emr_rotation(et, system)
        assert_allclose(np.linalg.det(C), 1.0, atol=1e-10)
        assert_allclose(C @ C.T, np.eye(3), atol=1e-10)

    def test_rotation_varies_with_time(self, system):
        """不同时间的旋转矩阵应不同。"""
        from e2m2e.core.rho_bridge import compute_emr_rotation

        et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
        et1 = et0 + 86400.0  # 一天后
        C0, _ = compute_emr_rotation(et0, system)
        C1, _ = compute_emr_rotation(et1, system)
        assert not np.allclose(C0, C1, atol=1e-6)

    def test_cdot_finite(self, system):
        """Cdot 应为有限值。"""
        from e2m2e.core.rho_bridge import compute_emr_rotation

        et = system.spice.utc_to_et("2025-06-21T11:00:06")
        _, Cdot = compute_emr_rotation(et, system)
        assert np.all(np.isfinite(Cdot))


# =============================================================================
# Test 往返一致性
# =============================================================================


class TestRoundTrip:
    """rho → ECI → rho 往返测试。"""

    def test_round_trip_zero_state(self, system, context):
        """零状态往返应精确恢复。"""
        from e2m2e.core.rho_bridge import eci_to_rho, rho_to_eci

        rho = np.zeros(3)
        rhodot = np.zeros(3)
        t_nd = 0.0

        r_eci, v_eci = rho_to_eci(rho, rhodot, t_nd, context, system)
        rho_back, rhodot_back = eci_to_rho(r_eci, v_eci, t_nd, context, system)

        assert_allclose(rho_back, rho, atol=1e-12)
        assert_allclose(rhodot_back, rhodot, atol=1e-12)

    def test_round_trip_nonzero_state(self, system, context):
        """非零状态往返误差 < 1e-9。"""
        from e2m2e.core.rho_bridge import eci_to_rho, rho_to_eci

        rho = np.array([0.01, -0.005, 0.002])
        rhodot = np.array([0.001, 0.002, -0.0005])
        t_nd = 0.5

        r_eci, v_eci = rho_to_eci(rho, rhodot, t_nd, context, system)
        rho_back, rhodot_back = eci_to_rho(r_eci, v_eci, t_nd, context, system)

        assert_allclose(rho_back, rho, atol=1e-9)
        assert_allclose(rhodot_back, rhodot, atol=1e-9)

    def test_round_trip_l2(self, system, context_l2):
        """L2 平动点往返误差 < 1e-9。"""
        from e2m2e.core.rho_bridge import eci_to_rho, rho_to_eci

        rho = np.array([0.01, 0.005, -0.001])
        rhodot = np.array([-0.001, 0.003, 0.0002])
        t_nd = 1.0

        r_eci, v_eci = rho_to_eci(rho, rhodot, t_nd, context_l2, system)
        rho_back, rhodot_back = eci_to_rho(r_eci, v_eci, t_nd, context_l2, system)

        assert_allclose(rho_back, rho, atol=1e-9)
        assert_allclose(rhodot_back, rhodot, atol=1e-9)

    def test_round_trip_different_times(self, system, context):
        """多个时间点往返一致性。"""
        from e2m2e.core.rho_bridge import eci_to_rho, rho_to_eci

        rho = np.array([0.005, -0.003, 0.001])
        rhodot = np.array([0.0005, 0.001, -0.0003])

        for t_nd in [0.0, 0.25, 0.5, 1.0, 2.0]:
            r_eci, v_eci = rho_to_eci(rho, rhodot, t_nd, context, system)
            rho_back, rhodot_back = eci_to_rho(r_eci, v_eci, t_nd, context, system)
            assert_allclose(rho_back, rho, atol=1e-9, err_msg=f"t_nd={t_nd}")
            assert_allclose(rhodot_back, rhodot, atol=1e-9, err_msg=f"t_nd={t_nd}")


# =============================================================================
# Test 物理合理性
# =============================================================================


class TestPhysicalPlausibility:
    """ECI 状态的物理合理性检验。"""

    def test_position_distance_range(self, system, context):
        """ECI 位置距地心 ~300000-400000 km。"""
        from e2m2e.core.rho_bridge import rho_to_eci

        rho = np.array([0.01, 0.0, 0.0])
        rhodot = np.zeros(3)
        r_eci, _ = rho_to_eci(rho, rhodot, 0.0, context, system)
        dist = np.linalg.norm(r_eci)
        assert 250_000 < dist < 500_000, f"距离 {dist:.0f} km 超出合理范围"

    def test_velocity_magnitude(self, system, context):
        """ECI 速度应 ~1 km/s 量级。"""
        from e2m2e.core.rho_bridge import rho_to_eci

        rho = np.array([0.01, 0.0, 0.0])
        rhodot = np.array([0.001, 0.001, 0.0])
        _, v_eci = rho_to_eci(rho, rhodot, 0.0, context, system)
        v_mag = np.linalg.norm(v_eci)
        assert 0.1 < v_mag < 5.0, f"速度 {v_mag:.3f} km/s 超出合理范围"

    def test_stationary_at_libration_point_has_v_lp_velocity(self, system, context):
        """rho=0、rhodot=0（航天器"停在"平动点）时，ECI 速度应等于平动点的
        J2000 速度 v_LP，而非 v_LP 经旋转后的 C@v_LP。

        防回归：v_eci 公式末项曾误写 C@v_LP（v_LP 已是 J2000，不该再旋转），
        导致 72h 跨验证发散 12 万 km。
        """
        from e2m2e.core.rho_bridge import _compute_lp_state_j2000, _jd_to_et, rho_to_eci

        t_nd = 0.0
        jd = context.jd0 + t_nd * context.TU / 86400.0
        et = _jd_to_et(jd, system)
        _, v_lp = _compute_lp_state_j2000(et, context, system)

        _, v_eci = rho_to_eci(np.zeros(3), np.zeros(3), t_nd, context, system)
        assert_allclose(v_eci, v_lp, atol=1e-9)

    def test_zero_rho_near_moon_distance(self, system, context):
        """rho=0 时，ECI 位置应接近地月距离（L1 距地 ~83% 地月距离）。"""
        from e2m2e.core.rho_bridge import rho_to_eci

        r_eci, _ = rho_to_eci(np.zeros(3), np.zeros(3), 0.0, context, system)
        dist = np.linalg.norm(r_eci)
        # L1 距地约 326000 km（0.847 * 384748）
        assert 280_000 < dist < 380_000, f"L1 距地 {dist:.0f} km，超出合理范围"

    def test_l2_farther_than_l1(self, system, context, context_l2):
        """L2 距地应远于 L1。"""
        from e2m2e.core.rho_bridge import rho_to_eci

        r_l1, _ = rho_to_eci(np.zeros(3), np.zeros(3), 0.0, context, system)
        r_l2, _ = rho_to_eci(np.zeros(3), np.zeros(3), 0.0, context_l2, system)
        assert np.linalg.norm(r_l2) > np.linalg.norm(r_l1)
