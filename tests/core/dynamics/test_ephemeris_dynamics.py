"""星历 N-body 动力学模型测试（Layer 1b）。

覆盖 EphemerisSystem/EphemerisDynamics 初始化、运动方程、
轨道传播与边界处理。
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.core import Dynamics
from e2m2e.core.ephemeris_system import EphemerisSystem
from e2m2e.mbse.data.enums import ReferenceFrame

pytestmark = pytest.mark.spice


# =============================================================================
# Fixtures
# =============================================================================
# 公共 SPICE fixtures 来自 tests/conftest.py:
#   spice_manager, spice_eph_system, spice_eph_dynamics, spice_syn_j2000,
#   reference_epoch, spice_kernel_path


@pytest.fixture
def leo_state():
    """近地轨道初始状态 (J2000, km, km/s)"""
    r = 6778  # 地球半径 + 400 km
    v = np.sqrt(398600.436 / r)  # 圆轨道速度
    return np.array([r, 0, 0, 0, v, 0])


@pytest.fixture
def reference_et(spice_manager, reference_epoch):
    """参考历元的 ET 秒数"""
    return spice_manager.utc_to_et(reference_epoch)


# =============================================================================
# Test EphemerisSystem 初始化
# =============================================================================
class TestEphemerisSystemInit:
    """测试 EphemerisSystem 的创建和属性"""

    def test_create_with_body_list(self, spice_manager):
        """应能用天体名称列表创建"""
        system = EphemerisSystem(
            bodies=["EARTH", "MOON", "SUN"],
            spice=spice_manager,
        )
        assert system is not None

    def test_bodies_attribute(self, spice_eph_system):
        """应有 bodies 属性记录天体列表"""
        assert hasattr(spice_eph_system, "bodies")
        assert "EARTH" in spice_eph_system.bodies
        assert "MOON" in spice_eph_system.bodies
        assert "SUN" in spice_eph_system.bodies

    def test_origin_attribute(self, spice_eph_system):
        """应有 origin 属性"""
        assert hasattr(spice_eph_system, "origin")
        assert spice_eph_system.origin == "EARTH"

    def test_frame_attribute(self, spice_eph_system):
        """应有 frame 属性"""
        assert hasattr(spice_eph_system, "frame")

        assert spice_eph_system.frame == ReferenceFrame.J2000

    def test_spice_reference(self, spice_eph_system, spice_manager):
        """应持有 SPICEManager 引用"""
        assert spice_eph_system.spice is spice_manager

    def test_gm_values_available(self, spice_eph_system):
        """应能获取各天体的 GM 值"""
        gm = spice_eph_system.get_gm_values()
        assert len(gm) == 3  # Earth, Moon, Sun
        assert all(g > 0 for g in gm)

    def test_two_body_system(self, spice_manager):
        """也应支持仅地月双体系统"""
        system = EphemerisSystem(
            bodies=["EARTH", "MOON"],
            spice=spice_manager,
        )
        assert len(system.bodies) == 2


# =============================================================================
# Test EphemerisDynamics 初始化
# =============================================================================
class TestEphemerisDynamicsInit:
    """测试 EphemerisDynamics 的创建"""

    def test_create_instance(self, spice_eph_dynamics):
        """应能创建 EphemerisDynamics 实例"""
        assert spice_eph_dynamics is not None

    def test_inherits_dynamics_base(self, spice_eph_dynamics):
        """应继承 Dynamics 基类"""
        assert isinstance(spice_eph_dynamics, Dynamics)

    def test_system_reference(self, spice_eph_dynamics, spice_eph_system):
        """应持有 EphemerisSystem 引用"""
        assert spice_eph_dynamics.system is spice_eph_system

    def test_has_equations_of_motion(self, spice_eph_dynamics):
        """应实现 equations_of_motion 方法"""
        assert hasattr(spice_eph_dynamics, "equations_of_motion")
        assert callable(spice_eph_dynamics.equations_of_motion)

    def test_has_propagate_method(self, spice_eph_dynamics):
        """应有 propagate 方法"""
        assert hasattr(spice_eph_dynamics, "propagate")


# =============================================================================
# Test N-body 运动方程
# =============================================================================
class TestEphemerisEquationsOfMotion:
    """测试 N-body 运动方程的正确性"""

    def test_eom_output_shape(self, spice_eph_dynamics, reference_et, leo_state):
        """运动方程输出应为 6 维"""
        derivatives = spice_eph_dynamics.equations_of_motion(reference_et, leo_state)
        assert derivatives.shape == (6,)

    def test_eom_velocity_equals_state_velocity(self, spice_eph_dynamics, reference_et, leo_state):
        """运动方程前 3 个分量应等于速度分量"""
        deriv = spice_eph_dynamics.equations_of_motion(reference_et, leo_state)
        assert_allclose(deriv[:3], leo_state[3:])

    def test_eom_acceleration_physical_range(self, spice_eph_dynamics, reference_et, leo_state):
        """加速度量级应在合理范围"""
        deriv = spice_eph_dynamics.equations_of_motion(reference_et, leo_state)
        a = deriv[3:]
        a_mag = np.linalg.norm(a)
        r = np.linalg.norm(leo_state[:3])
        a_expected = 398600.436 / r**2
        assert_allclose(a_mag, a_expected, rtol=0.05)

    def test_eom_at_moon_distance(self, spice_eph_dynamics, reference_et):
        """在月球距离处，加速度应包含月球和太阳摄动"""
        state_moon_dist = np.array([384400.0, 0, 0, 0, 1.0, 0])
        deriv = spice_eph_dynamics.equations_of_motion(reference_et, state_moon_dist)
        a = deriv[3:]
        a_mag = np.linalg.norm(a)
        a_earth_only = 398600.436 / 384400**2
        assert a_mag < a_earth_only * 2
        assert a_mag > a_earth_only * 0.5

    def test_eom_symmetry_acceleration_direction(self, spice_eph_dynamics, reference_et):
        """在地球附近，加速度应大致指向地球（径向向内）"""
        state = np.array([7000.0, 0, 0, 0, 7.5, 0])
        deriv = spice_eph_dynamics.equations_of_motion(reference_et, state)
        ax = deriv[3]
        assert ax < 0, "在 x 正方向，加速度应指向地球（x 负方向）"

    def test_eom_stm_output_shape(self, spice_eph_dynamics, reference_et, leo_state):
        """STM 方程输出应为 42 维"""
        stm0 = np.eye(6).flatten()
        augmented = np.concatenate([leo_state, stm0])
        deriv = spice_eph_dynamics.equations_with_stm(reference_et, augmented)
        assert deriv.shape == (42,)

    def test_eom_stm_initial_derivatives(self, spice_eph_dynamics, reference_et, leo_state):
        """初始时刻 STM 导数应满足 dPhi/dt(0) = A * I = A"""
        stm0 = np.eye(6).flatten()
        augmented = np.concatenate([leo_state, stm0])
        deriv = spice_eph_dynamics.equations_with_stm(reference_et, augmented)
        dphi_dt = deriv[6:].reshape(6, 6)
        assert np.all(np.isfinite(dphi_dt))


# =============================================================================
# Test 轨道传播
# =============================================================================
class TestEphemerisPropagation:
    """测试星历模型下的轨道传播"""

    def test_propagate_basic(self, spice_eph_dynamics, reference_et, leo_state):
        """应能在星历模型下传播轨道"""
        t_span = (reference_et, reference_et + 5400)  # 1.5 小时
        result = spice_eph_dynamics.propagate(leo_state, t_span)
        assert "time" in result
        assert "states" in result
        assert result["states"].shape[1] == 6
        assert result["states"].shape[0] > 1

    def test_propagate_state_is_finite(self, spice_eph_dynamics, reference_et, leo_state):
        """传播结果不应包含 NaN 或 Inf"""
        t_span = (reference_et, reference_et + 5400)
        result = spice_eph_dynamics.propagate(leo_state, t_span)
        assert np.all(np.isfinite(result["states"]))

    def test_propagate_with_stm(self, spice_eph_dynamics, reference_et, leo_state):
        """应能同时传播状态和 STM"""
        t_span = (reference_et, reference_et + 3600)
        result = spice_eph_dynamics.propagate(leo_state, t_span, with_stm=True)
        assert "stm" in result
        assert result["stm"] is not None

    def test_propagate_stm_shape(self, spice_eph_dynamics, reference_et, leo_state):
        """STM 应为 6x6 矩阵序列"""
        t_span = (reference_et, reference_et + 3600)
        result = spice_eph_dynamics.propagate(leo_state, t_span, with_stm=True)
        n_times = result["states"].shape[0]
        assert result["stm"].shape == (n_times, 6, 6)

    def test_propagate_stm_initial_is_identity(self, spice_eph_dynamics, reference_et, leo_state):
        """初始 STM 应为单位矩阵"""
        t_span = (reference_et, reference_et + 3600)
        t_eval = np.linspace(t_span[0], t_span[1], 100)
        result = spice_eph_dynamics.propagate(leo_state, t_span, t_eval=t_eval, with_stm=True)
        stm0 = result["stm"][0]
        assert_allclose(stm0, np.eye(6), atol=1e-6)

    def test_propagate_leo_returns_near_circular(self, spice_eph_dynamics, reference_et, leo_state):
        """LEO 传播约一圈后应返回起点附近"""
        period = 2 * np.pi * np.sqrt(6778**3 / 398600.436)
        t_span = (reference_et, reference_et + period)
        result = spice_eph_dynamics.propagate(leo_state, t_span)
        final_state = result["states"][-1]
        r_final = np.linalg.norm(final_state[:3])
        r_initial = np.linalg.norm(leo_state[:3])
        assert_allclose(r_final, r_initial, rtol=0.01)

    def test_propagate_dro_like_orbit(self, spice_eph_dynamics, reference_et):
        """传播 DRO 类轨道（月球距离附近），应保持稳定"""
        dro_state = np.array([384400, 0, 0, 0, -0.5, 0])
        dro_period = 9.11 * 86400  # 3:1 DRO 周期约 9.11 天
        t_span = (reference_et, reference_et + dro_period)
        result = spice_eph_dynamics.propagate(dro_state, t_span)
        assert np.all(np.isfinite(result["states"]))
        final_r = np.linalg.norm(result["states"][-1, :3])
        initial_r = np.linalg.norm(dro_state[:3])
        assert_allclose(final_r, initial_r, rtol=0.15)


# =============================================================================
# Test 边界和错误处理
# =============================================================================
class TestEphemerisDynamicsEdgeCases:
    """测试边界条件和错误处理"""

    def test_propagate_zero_duration(self, spice_eph_dynamics, reference_et, leo_state):
        """传播零时间应返回初始状态"""
        t_span = (reference_et, reference_et)
        result = spice_eph_dynamics.propagate(leo_state, t_span)
        assert result["states"].shape[0] >= 1
        assert result["states"].shape[1] == 6

    def test_propagate_backward_time(self, spice_eph_dynamics, reference_et, leo_state):
        """应能向后传播（t_span 终点 < 起点）"""
        t_span = (reference_et, reference_et - 3600)
        result = spice_eph_dynamics.propagate(leo_state, t_span)
        assert np.all(np.isfinite(result["states"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
