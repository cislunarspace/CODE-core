"""FiniteBurn direction_frame 测试（TDD）。

验证 VNB/LVLH 坐标系转换、非法帧拒绝、
callable 方向与零速度/零位置边界。

低推力功能尚未开发完成，FiniteBurn 的 Rust 传播路径暂缺；本文件标记
``low_thrust``，本轮检查排除在绿门外。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import FiniteBurn

pytestmark = [pytest.mark.force, pytest.mark.low_thrust]


# --- 构造辅助 ---


def _make_state(r, v):
    """构造 6 维状态向量。"""
    return np.array([*r, *v], dtype=float)


class _FakeSystem:
    """最小 System 桩。"""

    coordinate_system = object()

    def gravitational_parameter(self, body):
        return 398600.4415


# --- direction_frame 参数校验 ---


def test_finite_burn_invalid_direction_frame_raises():
    """非法 direction_frame 在构造时抛 ValueError。"""
    with pytest.raises(ValueError, match="direction_frame"):
        FiniteBurn(
            thrust_profile=lambda t: 10.0,
            direction=[1.0, 0.0, 0.0],
            mass=1000.0,
            direction_frame="INVALID",
        )


def test_finite_burn_vnb_direction_frame_accepted():
    """direction_frame='VNB' 合法。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[1.0, 0.0, 0.0],
        mass=1000.0,
        direction_frame="VNB",
    )
    assert burn.direction_frame == "VNB"


def test_finite_burn_lvlh_direction_frame_accepted():
    """direction_frame='LVLH' 合法。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[0.0, 1.0, 0.0],
        mass=1000.0,
        direction_frame="LVLH",
    )
    assert burn.direction_frame == "LVLH"


def test_finite_burn_none_direction_frame_default():
    """direction_frame=None 为默认值，保持原有行为。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[1.0, 0.0, 0.0],
        mass=1000.0,
    )
    assert burn.direction_frame is None


# --- direction_frame=None 时保持原有行为 ---


def test_finite_burn_none_frame_fixed_direction():
    """direction_frame=None 时，固定方向直接归一化使用。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[2.0, 0.0, 0.0],  # 非单位向量
        mass=1000.0,
    )
    state = _make_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    acc = burn.compute_acceleration(0.0, state, _FakeSystem())

    # 10N / 1000kg = 0.01 m/s² = 1e-5 km/s²，方向 [1,0,0]
    np.testing.assert_allclose(acc, [1e-5, 0.0, 0.0])


def test_finite_burn_none_frame_callable_direction():
    """direction_frame=None 时，callable 返回值直接归一化使用。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=lambda t, state: [0.0, 3.0, 0.0],
        mass=1000.0,
    )
    state = _make_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    acc = burn.compute_acceleration(0.0, state, _FakeSystem())

    np.testing.assert_allclose(acc, [0.0, 1e-5, 0.0])


# --- VNB 坐标系转换 ---


def test_finite_burn_vnb_velocity_direction():
    """VNB 下 direction=[1,0,0] 对应速度方向（V）。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[1.0, 0.0, 0.0],
        mass=1000.0,
        direction_frame="VNB",
    )
    # 状态：r=[7000,0,0], v=[0,7.5,0] → V 方向为 [0,1,0]
    state = _make_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    acc = burn.compute_acceleration(0.0, state, _FakeSystem())

    # 推力沿 V 方向 = [0,1,0]，大小 1e-5 km/s²
    np.testing.assert_allclose(acc, [0.0, 1e-5, 0.0], atol=1e-12)


def test_finite_burn_vnb_normal_direction():
    """VNB 下 direction=[0,1,0] 对应角动量方向（N = r × v / |r × v|）。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[0.0, 1.0, 0.0],
        mass=1000.0,
        direction_frame="VNB",
    )
    # r=[7000,0,0], v=[0,7.5,0] → r×v = [0,0,52500] → N = [0,0,1]
    state = _make_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    acc = burn.compute_acceleration(0.0, state, _FakeSystem())

    np.testing.assert_allclose(acc, [0.0, 0.0, 1e-5], atol=1e-12)


def test_finite_burn_vnb_binormal_direction():
    """VNB 下 direction=[0,0,1] 对应 B = V × N 方向。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[0.0, 0.0, 1.0],
        mass=1000.0,
        direction_frame="VNB",
    )
    # r=[7000,0,0], v=[0,7.5,0] → V=[0,1,0], N=[0,0,1], B=V×N=[1,0,0]
    state = _make_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    acc = burn.compute_acceleration(0.0, state, _FakeSystem())

    np.testing.assert_allclose(acc, [1e-5, 0.0, 0.0], atol=1e-12)


def test_finite_burn_vnb_combined_direction():
    """VNB 下 direction=[1,1,1] 产生混合方向。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[1.0, 1.0, 1.0],
        mass=1000.0,
        direction_frame="VNB",
    )
    # r=[7000,0,0], v=[0,7.5,0] → V=[0,1,0], N=[0,0,1], B=[1,0,0]
    # direction=[1,1,1] 在 VNB 下 = V + N + B = [0,1,0] + [0,0,1] + [1,0,0] = [1,1,1]
    # 归一化后 = [1,1,1]/sqrt(3)
    state = _make_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    acc = burn.compute_acceleration(0.0, state, _FakeSystem())

    expected_dir = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    expected = 1e-5 * expected_dir
    np.testing.assert_allclose(acc, expected, atol=1e-12)


# --- LVLH 坐标系转换 ---


def test_finite_burn_lvlh_radial_direction():
    """LVLH 下 direction=[1,0,0] 对应径向（R = r/|r|）。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[1.0, 0.0, 0.0],
        mass=1000.0,
        direction_frame="LVLH",
    )
    # r=[7000,0,0] → R=[1,0,0]
    state = _make_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    acc = burn.compute_acceleration(0.0, state, _FakeSystem())

    np.testing.assert_allclose(acc, [1e-5, 0.0, 0.0], atol=1e-12)


def test_finite_burn_lvlh_velocity_direction():
    """LVLH 下 direction=[0,1,0] 对应沿迹方向（V = v/|v|）。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[0.0, 1.0, 0.0],
        mass=1000.0,
        direction_frame="LVLH",
    )
    # v=[0,7.5,0] → V=[0,1,0]
    state = _make_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    acc = burn.compute_acceleration(0.0, state, _FakeSystem())

    np.testing.assert_allclose(acc, [0.0, 1e-5, 0.0], atol=1e-12)


def test_finite_burn_lvlh_cross_track_direction():
    """LVLH 下 direction=[0,0,1] 对应轨道面法向（N = R × V）。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[0.0, 0.0, 1.0],
        mass=1000.0,
        direction_frame="LVLH",
    )
    # r=[7000,0,0], v=[0,7.5,0] → R=[1,0,0], V=[0,1,0], N=R×V=[0,0,1]
    state = _make_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    acc = burn.compute_acceleration(0.0, state, _FakeSystem())

    np.testing.assert_allclose(acc, [0.0, 0.0, 1e-5], atol=1e-12)


def test_finite_burn_lvlh_3d_position():
    """LVLH 在三维非共面位置下正确。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[1.0, 0.0, 0.0],
        mass=1000.0,
        direction_frame="LVLH",
    )
    # r=[1000,2000,3000], v=[1,2,3]
    state = _make_state([1000.0, 2000.0, 3000.0], [1.0, 2.0, 3.0])
    acc = burn.compute_acceleration(0.0, state, _FakeSystem())

    r = np.array([1000.0, 2000.0, 3000.0])
    r_hat = r / np.linalg.norm(r)
    expected = 1e-5 * r_hat
    np.testing.assert_allclose(acc, expected, atol=1e-12)


# --- callable direction + direction_frame ---


def test_finite_burn_vnb_with_callable_direction():
    """direction 为 callable 时，返回值在 direction_frame 下解释。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=lambda t, state: [1.0, 0.0, 0.0],  # VNB 下 = V 方向
        mass=1000.0,
        direction_frame="VNB",
    )
    state = _make_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    acc = burn.compute_acceleration(0.0, state, _FakeSystem())

    # V 方向 = [0,1,0]
    np.testing.assert_allclose(acc, [0.0, 1e-5, 0.0], atol=1e-12)


def test_finite_burn_lvlh_with_callable_direction():
    """direction callable 在 LVLH 下解释。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=lambda t, state: [0.0, 0.0, 1.0],  # LVLH 下 = N 方向
        mass=1000.0,
        direction_frame="LVLH",
    )
    state = _make_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    acc = burn.compute_acceleration(0.0, state, _FakeSystem())

    # N = R × V = [0,0,1]
    np.testing.assert_allclose(acc, [0.0, 0.0, 1e-5], atol=1e-12)


# --- 零速度/零位置边界 ---


def test_finite_burn_vnb_zero_velocity_raises():
    """VNB 下 |v|=0 时无法构造 V 方向，抛 ValueError。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[1.0, 0.0, 0.0],
        mass=1000.0,
        direction_frame="VNB",
    )
    state = _make_state([7000.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="velocity"):
        burn.compute_acceleration(0.0, state, _FakeSystem())


def test_finite_burn_lvlh_zero_position_raises():
    """LVLH 下 |r|=0 时无法构造 R 方向，抛 ValueError。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 10.0,
        direction=[1.0, 0.0, 0.0],
        mass=1000.0,
        direction_frame="LVLH",
    )
    state = _make_state([0.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    with pytest.raises(ValueError, match="position"):
        burn.compute_acceleration(0.0, state, _FakeSystem())


# --- 关机时 direction_frame 不触发计算 ---


def test_finite_burn_zero_thrust_skips_direction_frame():
    """thrust=0 时直接返回零，不解析 direction_frame。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 0.0,
        direction=[1.0, 0.0, 0.0],
        mass=1000.0,
        direction_frame="VNB",
    )
    # 即使 v=0（VNB 会抛），thrust=0 也应直接返回零
    state = _make_state([7000.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    acc = burn.compute_acceleration(0.0, state, _FakeSystem())

    np.testing.assert_array_equal(acc, np.zeros(3))
