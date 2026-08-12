"""ImpulsiveBurn / FiniteBurn 单元测试。

覆盖冻结拷贝、零推力、常值推力、固定/可调用方向与归一化。

低推力功能尚未开发完成（Facade 任务入口占位，Rust 传播路径已接入）；本文件标记
``low_thrust``，本轮检查排除在绿门外。
"""

import dataclasses

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel
from e2m2e.algorithm.forces.thrust import FiniteBurn, ImpulsiveBurn

pytestmark = [pytest.mark.force, pytest.mark.low_thrust]


class _FakeSystem:
    """仅用于传播测试的最小 System 桩。"""

    def __init__(self):
        self.coordinate_system = object()

    @property
    def frame(self):
        from e2m2e.mbse.data.enums import ReferenceFrame

        return ReferenceFrame.J2000

    @property
    def unit_system(self):
        from e2m2e.mbse.data.enums import UnitSystem

        return UnitSystem.SI

    def gravitational_parameter(self, body):
        return 398600.4415


def test_impulsive_burn_stores_copied_delta_v_and_is_frozen():
    """ImpulsiveBurn 存 epoch + delta_v（拷贝），且 frozen 不可变。"""
    dv = np.array([0.1, 0.2, 0.3])
    burn = ImpulsiveBurn(epoch=1.0, delta_v=dv)

    assert burn.epoch == 1.0
    np.testing.assert_allclose(burn.delta_v, [0.1, 0.2, 0.3])

    # 存储为拷贝：修改原数组不影响 burn
    dv[0] = 99.0
    np.testing.assert_allclose(burn.delta_v, [0.1, 0.2, 0.3])

    # frozen：不可重新赋值
    with pytest.raises(dataclasses.FrozenInstanceError):
        burn.epoch = 2.0


def test_finite_burn_zero_thrust_returns_zero_acceleration():
    """thrust_profile 返回 0 → compute_acceleration 返回 zeros(3)。"""
    burn = FiniteBurn(
        thrust_profile=lambda t: 0.0,
        direction=np.array([1.0, 0.0, 0.0]),
        mass=1000.0,
    )
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    acc = burn.compute_acceleration(0.0, state, system=None)
    np.testing.assert_allclose(acc, np.zeros(3), atol=1e-15)


def test_finite_burn_constant_thrust_fixed_direction():
    """常值推力 + 固定方向 → a = thrust/mass/1000 · d̂（内部归一化方向）。"""
    state = np.zeros(6)

    # 轴对齐：方向 [3,0,0] 归一化为 [1,0,0]
    burn = FiniteBurn(
        thrust_profile=lambda t: 500.0,
        direction=np.array([3.0, 0.0, 0.0]),
        mass=1000.0,
    )
    acc = burn.compute_acceleration(0.0, state, system=None)
    # 500 N / 1000 kg = 0.5 m/s² → 5e-4 km/s²
    np.testing.assert_allclose(acc, [5e-4, 0.0, 0.0], rtol=1e-12)

    # 非轴对齐：验证归一化
    burn2 = FiniteBurn(
        thrust_profile=lambda t: 500.0,
        direction=np.array([1.0, 1.0, 0.0]),
        mass=1000.0,
    )
    acc2 = burn2.compute_acceleration(0.0, state, system=None)
    expected = 0.5 / 1000.0 * np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(acc2, expected, rtol=1e-12)


def test_finite_burn_callable_direction_normalized():
    """可调用方向 (t, state) -> (3,) 被调用并归一化使用（如沿速度方向）。"""
    velocity = np.array([0.0, 7.5, 0.0])
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    burn = FiniteBurn(
        thrust_profile=lambda t: 500.0,
        direction=lambda t, s: s[3:6],  # 返回速度向量（未归一化）
        mass=1000.0,
    )
    acc = burn.compute_acceleration(0.0, state, system=None)
    expected = 0.5 / 1000.0 * velocity / np.linalg.norm(velocity)
    np.testing.assert_allclose(acc, expected, rtol=1e-12)


def test_finite_burn_validation():
    """mass≤0 / 固定方向零向量 → 构造时 ValueError；thrust<0 → 求值时 ValueError。"""
    # mass ≤ 0
    with pytest.raises(ValueError, match="mass"):
        FiniteBurn(lambda t: 1.0, np.array([1.0, 0.0, 0.0]), 0.0)
    with pytest.raises(ValueError, match="mass"):
        FiniteBurn(lambda t: 1.0, np.array([1.0, 0.0, 0.0]), -1.0)

    # 固定方向零向量
    with pytest.raises(ValueError, match="direction"):
        FiniteBurn(lambda t: 1.0, np.zeros(3), 1000.0)

    # thrust < 0（求值时）
    burn = FiniteBurn(lambda t: -5.0, np.array([1.0, 0.0, 0.0]), 1000.0)
    with pytest.raises(ValueError, match="thrust"):
        burn.compute_acceleration(0.0, np.zeros(6), system=None)


def test_finite_burn_spiral_raises_semi_major_axis(point_mass_force):
    """沿速度方向低推力传播 → 半长轴持续提升（电推进螺旋趋势）。"""
    mu = 398600.4415
    r0 = 6678.0  # LEO 300km
    v_circ = np.sqrt(mu / r0)
    y0 = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0])

    burn = FiniteBurn(
        thrust_profile=lambda t: 2.0,  # N
        direction=lambda t, s: s[3:6],  # 沿速度方向（可调用）
        mass=1000.0,  # kg
    )
    fm = ForceModel(_FakeSystem(), forces=[point_mass_force, burn])

    period = 2.0 * np.pi * np.sqrt(r0**3 / mu)
    n_orbits = 3
    result = fm.propagate(
        y0, (0.0, n_orbits * period), t_eval=np.linspace(0.0, n_orbits * period, 60)
    )

    states = result["states"]
    r = np.linalg.norm(states[:, :3], axis=1)
    v = np.linalg.norm(states[:, 3:], axis=1)
    energy = 0.5 * v**2 - mu / r
    semi_major = -mu / (2.0 * energy)

    # 半长轴末端 > 起点
    assert semi_major[-1] > semi_major[0]
    # 整体趋势向上（线性拟合斜率 > 0）
    slope = np.polyfit(np.arange(semi_major.size), semi_major, 1)[0]
    assert slope > 0.0
