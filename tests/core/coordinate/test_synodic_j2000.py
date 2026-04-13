"""
需求: CR3BP 旋转系 ↔ J2000 惯性系 坐标转换 (Layer 1c)

e2m2e 需要实现 CR3BP 旋转系 (synodic frame) 与 J2000 惯性系之间的坐标转换，
以支持将 CR3BP 中计算的轨道转换到高精度星历模型。

功能要求:
  1. synodic → J2000: 将 CR3BP 旋转系状态转换到 J2000 惯性系
  2. J2000 → synodic: 将 J2000 惯性系状态转换到 CR3BP 旋转系
  3. 使用 SPICE 获取瞬时旋转矩阵（基于天体实际位置）
  4. 速度转换包含 Coriolis 项
  5. 支持批量转换（多个时间点）

转换算法 (synodic → J2000):
  1. 将无量纲 CR3BP 时间转换为 ET: t_et = et0 + t_syn * t_c
  2. 用 SPICE 获取月球在 J2000 下的位置 r_m2 和速度 v_m2
  3. 计算瞬时特征长度 l_c = |r_m2|
  4. 构造旋转矩阵:
     e1 = r_m2 / |r_m2|  (x 轴指向月球)
     e3 = (r_m2 × v_m2) / |r_m2 × v_m2|  (z 轴为轨道法线)
     e2 = e3 × e1  (y 轴完成右手系)
  5. 有量纲化: r_dim = r_syn * l_c, v_dim = v_syn * l_c / t_c
  6. 原点平移（如果 CR3BP 以某天体为中心）
  7. 旋转: state_j2000 = R @ state_dim

参考实现:
  SEMpy synodic_j2000.py 中的 synodic_to_j2000 和 j2000_to_synodic
  注意: SEMpy 的 CR3BP 以质心为原点, Moon 在 (1-mu, 0, 0) 处

依赖:
  Layer 1a (SPICEManager)
  CR3BP_System (已有)
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from e2m2e.core import (
    CR3BP_System,
    SPICEManager,
    SynodicJ2000Transformation,
)

pytestmark = pytest.mark.spice


# =============================================================================
# Fixtures
# =============================================================================
from tests.conftest import MU, DU, TU_SECONDS


@pytest.fixture
def cr3bp_system():
    """创建地月 CR3BP 系统"""
    return CR3BP_System(mu=MU, primary="earth", secondary="moon")


@pytest.fixture
def spice_manager(spice_kernel_path):
    """加载了 DE440 的 SPICEManager"""
    mgr = SPICEManager()
    mgr.load_kernel(spice_kernel_path)
    yield mgr
    mgr.unload_kernel(spice_kernel_path)


@pytest.fixture
def syn_j2000(cr3bp_system, spice_manager):
    """创建 synodic ↔ J2000 坐标转换器"""
    return SynodicJ2000Transformation(
        cr3bp_system=cr3bp_system,
        spice=spice_manager,
    )


@pytest.fixture
def reference_et(spice_manager, reference_epoch):
    """参考历元 ET"""
    return spice_manager.utc_to_et(reference_epoch)


@pytest.fixture
def dro_synodic_state():
    """3:1 DRO 初始状态 (CR3BP 旋转系, 无量纲)"""
    return np.array([1.1202109158830986, 0.0, 0.0, 0.0, -0.46178983697629084, 0.0])


# =============================================================================
# Test SynodicJ2000Transformation 初始化
# =============================================================================
class TestSynodicJ2000Init:
    """测试坐标转换器的创建"""

    def test_create_instance(self, cr3bp_system, spice_manager):
        """应能创建 SynodicJ2000Transformation"""
        transform = SynodicJ2000Transformation(
            cr3bp_system=cr3bp_system,
            spice=spice_manager,
        )
        assert transform is not None

    def test_has_synodic_to_j2000_method(self, syn_j2000):
        """应有 synodic_to_j2000 方法"""
        assert hasattr(syn_j2000, "synodic_to_j2000")
        assert callable(syn_j2000.synodic_to_j2000)

    def test_has_j2000_to_synodic_method(self, syn_j2000):
        """应有 j2000_to_synodic 方法"""
        assert hasattr(syn_j2000, "j2000_to_synodic")
        assert callable(syn_j2000.j2000_to_synodic)

    def test_system_reference(self, syn_j2000, cr3bp_system):
        """应持有 CR3BP 系统引用"""
        assert syn_j2000.cr3bp_system is cr3bp_system

    def test_spice_reference(self, syn_j2000, spice_manager):
        """应持有 SPICEManager 引用"""
        assert syn_j2000.spice is spice_manager


# =============================================================================
# Test synodic → J2000 转换
# =============================================================================
class TestSynodicToJ2000:
    """测试 synodic → J2000 坐标转换"""

    def test_output_shape(self, syn_j2000, reference_et, dro_synodic_state):
        """输出应为 6 维状态向量"""
        state_j2000 = syn_j2000.synodic_to_j2000(
            state_syn=dro_synodic_state,
            t_syn=0.0,
            et0=reference_et,
        )
        assert state_j2000.shape == (6,)

    def test_output_is_finite(self, syn_j2000, reference_et, dro_synodic_state):
        """输出不应包含 NaN 或 Inf"""
        state_j2000 = syn_j2000.synodic_to_j2000(
            state_syn=dro_synodic_state,
            t_syn=0.0,
            et0=reference_et,
        )
        assert np.all(np.isfinite(state_j2000))

    def test_position_physical_range(self, syn_j2000, reference_et, dro_synodic_state):
        """DRO 在 J2000 下的位置应在月球距离附近"""
        state_j2000 = syn_j2000.synodic_to_j2000(
            state_syn=dro_synodic_state,
            t_syn=0.0,
            et0=reference_et,
        )
        r_from_earth = np.linalg.norm(state_j2000[:3])
        assert 300000 < r_from_earth < 500000, (
            f"DRO距地球 {r_from_earth:.0f} km，超出合理范围"
        )

    def test_velocity_physical_range(self, syn_j2000, reference_et, dro_synodic_state):
        """DRO 在 J2000 下的速度应合理"""
        state_j2000 = syn_j2000.synodic_to_j2000(
            state_syn=dro_synodic_state,
            t_syn=0.0,
            et0=reference_et,
        )
        v = np.linalg.norm(state_j2000[3:])
        assert 0.01 < v < 5.0, f"DRO速度 {v:.3f} km/s，超出合理范围"

    def test_origin_at_moon_position(self, syn_j2000, reference_et):
        """CR3BP 中月球位置 (1-mu, 0, 0) 转换到 J2000 后应接近 SPICE 月球位置"""
        mu = MU
        moon_synodic = np.array([1 - mu, 0, 0, 0, 0, 0])
        state_j2000 = syn_j2000.synodic_to_j2000(
            state_syn=moon_synodic,
            t_syn=0.0,
            et0=reference_et,
        )
        spice_moon = syn_j2000.spice.get_body_state(
            target="MOON", et=reference_et, frame="J2000", observer="EARTH"
        )
        r_transform = np.linalg.norm(state_j2000[:3])
        r_spice = np.linalg.norm(spice_moon[:3])
        assert_allclose(r_transform, r_spice, rtol=0.01)

    def test_different_times_different_positions(self, syn_j2000, reference_et, dro_synodic_state):
        """不同 CR3BP 时刻转换到 J2000 应给出不同位置"""
        tc = TU_SECONDS
        state_t0 = syn_j2000.synodic_to_j2000(
            state_syn=dro_synodic_state, t_syn=0.0, et0=reference_et
        )
        state_t1 = syn_j2000.synodic_to_j2000(
            state_syn=dro_synodic_state, t_syn=1.0, et0=reference_et
        )
        assert not np.allclose(state_t0[:3], state_t1[:3])


# =============================================================================
# Test J2000 → synodic 转换
# =============================================================================
class TestJ2000ToSynodic:
    """测试 J2000 → synodic 坐标转换"""

    def test_output_shape(self, syn_j2000, reference_et, dro_synodic_state):
        """输出应为 6 维状态向量"""
        state_j2000 = syn_j2000.synodic_to_j2000(
            state_syn=dro_synodic_state,
            t_syn=0.0,
            et0=reference_et,
        )
        state_back = syn_j2000.j2000_to_synodic(
            state_j2000=state_j2000,
            t_syn=0.0,
            et0=reference_et,
        )
        assert state_back.shape == (6,)

    def test_output_is_finite(self, syn_j2000, reference_et, dro_synodic_state):
        """输出不应包含 NaN 或 Inf"""
        state_j2000 = syn_j2000.synodic_to_j2000(
            state_syn=dro_synodic_state,
            t_syn=0.0,
            et0=reference_et,
        )
        state_back = syn_j2000.j2000_to_synodic(
            state_j2000=state_j2000,
            t_syn=0.0,
            et0=reference_et,
        )
        assert np.all(np.isfinite(state_back))


# =============================================================================
# Test 往返转换 (Round-trip)
# =============================================================================
class TestSynodicJ2000RoundTrip:
    """测试正向+反向转换应恢复原始状态"""

    def test_round_trip_at_t0(self, syn_j2000, reference_et, dro_synodic_state):
        """synodic → J2000 → synodic 在 t=0 应恢复原始状态"""
        state_j2000 = syn_j2000.synodic_to_j2000(
            state_syn=dro_synodic_state, t_syn=0.0, et0=reference_et
        )
        state_back = syn_j2000.j2000_to_synodic(
            state_j2000=state_j2000, t_syn=0.0, et0=reference_et
        )
        assert_allclose(state_back, dro_synodic_state, atol=1e-8)

    def test_round_trip_at_nonzero_time(self, syn_j2000, reference_et, dro_synodic_state):
        """synodic → J2000 → synodic 在 t≠0 应恢复原始状态"""
        t_syn = 1.0475  # 约半 DRO 周期
        state_j2000 = syn_j2000.synodic_to_j2000(
            state_syn=dro_synodic_state, t_syn=t_syn, et0=reference_et
        )
        state_back = syn_j2000.j2000_to_synodic(
            state_j2000=state_j2000, t_syn=t_syn, et0=reference_et
        )
        assert_allclose(state_back, dro_synodic_state, atol=1e-8)

    def test_inverse_round_trip(self, syn_j2000, reference_et):
        """J2000 → synodic → J2000 应恢复原始状态"""
        spice_state = syn_j2000.spice.get_body_state(
            target="MOON", et=reference_et, frame="J2000", observer="EARTH"
        )
        state_syn = syn_j2000.j2000_to_synodic(
            state_j2000=spice_state, t_syn=0.0, et0=reference_et
        )
        state_back = syn_j2000.synodic_to_j2000(
            state_syn=state_syn, t_syn=0.0, et0=reference_et
        )
        assert_allclose(state_back, spice_state, atol=1e-3)


# =============================================================================
# Test 批量转换
# =============================================================================
class TestBatchConversion:
    """测试批量坐标转换"""

    def test_batch_synodic_to_j2000(self, syn_j2000, reference_et, dro_synodic_state):
        """应能批量转换多个时间点的状态"""
        tc = TU_SECONDS
        t_syn_arr = np.linspace(0, 2.095, 50)
        states_syn = np.array([dro_synodic_state] * 50)

        states_j2000 = syn_j2000.batch_synodic_to_j2000(
            states_syn=states_syn,
            t_syn_arr=t_syn_arr,
            et0=reference_et,
        )
        assert states_j2000.shape == (50, 6)
        assert np.all(np.isfinite(states_j2000))

    def test_batch_round_trip(self, syn_j2000, reference_et, dro_synodic_state):
        """批量 synodic → J2000 → synodic 应恢复"""
        t_syn_arr = np.linspace(0, 1.0, 20)
        states_syn = np.array([dro_synodic_state] * 20)

        states_j2000 = syn_j2000.batch_synodic_to_j2000(
            states_syn=states_syn,
            t_syn_arr=t_syn_arr,
            et0=reference_et,
        )
        states_back = syn_j2000.batch_j2000_to_synodic(
            states_j2000=states_j2000,
            t_syn_arr=t_syn_arr,
            et0=reference_et,
        )
        assert_allclose(states_back, states_syn, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
