"""PointMassGravity 测试（TDD）。

验证显式 mu、system 回退与月球体切换。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import PointMassGravity


class _FakeSystem:
    """最小 System 桩，提供 gravitational_parameter。"""

    def gravitational_parameter(self, body):
        if body.upper() == "EARTH":
            return 398600.4415
        if body.upper() == "MOON":
            return 4902.800122
        raise ValueError(f"unknown body {body}")


def test_point_mass_gravity_with_explicit_mu():
    """显式传入 mu 时，从 mu 计算加速度。"""
    force = PointMassGravity(body="EARTH", mu=398600.4415)
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    acc = force.compute_acceleration(0.0, state, None)

    r = 7000.0
    expected = -398600.4415 / (r**3) * np.array([r, 0.0, 0.0])
    np.testing.assert_allclose(acc, expected)


def test_point_mass_gravity_falls_back_to_system():
    """mu=None 时从 system.gravitational_parameter(body) 获取。"""
    system = _FakeSystem()
    force = PointMassGravity(body="EARTH")
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    acc = force.compute_acceleration(0.0, state, system)

    r = 7000.0
    expected = -398600.4415 / (r**3) * np.array([r, 0.0, 0.0])
    np.testing.assert_allclose(acc, expected)


def test_point_mass_gravity_moon_body():
    """换用 MOON 体，从 system 获取对应 mu。"""
    system = _FakeSystem()
    force = PointMassGravity(body="MOON")
    state = np.array([2000.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    acc = force.compute_acceleration(0.0, state, system)

    r = 2000.0
    expected = -4902.800122 / (r**3) * np.array([r, 0.0, 0.0])
    np.testing.assert_allclose(acc, expected)


def test_point_mass_gravity_no_system_no_mu_raises():
    """mu=None 且 system=None 时抛 ValueError。"""
    force = PointMassGravity(body="EARTH")
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    with pytest.raises(ValueError, match="gravitational_parameter"):
        force.compute_acceleration(0.0, state, None)


def test_point_mass_gravity_zero_position():
    """r=0 时返回零向量（避免除零）。"""
    force = PointMassGravity(body="EARTH", mu=398600.4415)
    state = np.zeros(6)
    acc = force.compute_acceleration(0.0, state, None)
    np.testing.assert_array_equal(acc, np.zeros(3))


def test_point_mass_gravity_3d_position():
    """三维位置下的加速度方向与大小正确。"""
    force = PointMassGravity(body="EARTH", mu=398600.4415)
    state = np.array([1000.0, 2000.0, 3000.0, 1.0, 2.0, 3.0])
    acc = force.compute_acceleration(0.0, state, None)

    r = np.array([1000.0, 2000.0, 3000.0])
    r_norm = np.linalg.norm(r)
    expected = -398600.4415 / (r_norm**3) * r
    np.testing.assert_allclose(acc, expected)


def test_point_mass_gravity_is_physical_model():
    """PointMassGravity 是 PhysicalModel 子类。"""
    from e2m2e.algorithm.forces import PhysicalModel

    assert issubclass(PointMassGravity, PhysicalModel)
