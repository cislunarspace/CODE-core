"""IndirectTerm 测试。

验证显式 mu、system 回退、零位置保护与天体切换。IndirectTerm 的加速度
取摄动天体相对原点的位置（system.get_body_position），而非航天器位置，
测试桩据此构造。
"""

import numpy as np
import pytest

from e2m2e.core.forces import IndirectTerm


class _FakeSystem:
    """最小 System 桩，提供 gravitational_parameter 与 get_body_position。"""

    def gravitational_parameter(self, body):
        if body.upper() == "EARTH":
            return 398600.4415
        if body.upper() == "MOON":
            return 4902.800122
        raise ValueError(f"unknown body {body}")

    def get_body_position(self, body, t):
        """返回天体位置（km）。固定给出典型地月距离量级的值。"""
        if body.upper() == "MOON":
            return np.array([384000.0, 0.0, 0.0])
        if body.upper() == "EARTH":
            return np.array([0.0, 0.0, 0.0])
        raise ValueError(f"unknown body {body}")


def test_indirect_term_with_explicit_mu():
    """显式传入 mu 时，从 mu + 天体位置计算间接项加速度。"""
    system = _FakeSystem()
    force = IndirectTerm(body="MOON", mu=4902.800122)
    acc = force.compute_acceleration(0.0, np.zeros(6), system)

    r = 384000.0
    expected = -4902.800122 / (r**3) * np.array([r, 0.0, 0.0])
    np.testing.assert_allclose(acc, expected)


def test_indirect_term_falls_back_to_system():
    """mu=None 时从 system.gravitational_parameter(body) 获取。"""
    system = _FakeSystem()
    force = IndirectTerm(body="MOON")
    acc = force.compute_acceleration(0.0, np.zeros(6), system)

    r = 384000.0
    expected = -4902.800122 / (r**3) * np.array([r, 0.0, 0.0])
    np.testing.assert_allclose(acc, expected)


def test_indirect_term_earth_body():
    """换 EARTH 体（原点处 r=0）返回零向量。"""
    system = _FakeSystem()
    force = IndirectTerm(body="EARTH", mu=398600.4415)
    acc = force.compute_acceleration(0.0, np.zeros(6), system)
    np.testing.assert_array_equal(acc, np.zeros(3))


def test_indirect_term_no_system_no_mu_raises():
    """mu=None 且 system=None 时抛 ValueError。"""
    force = IndirectTerm(body="MOON")
    with pytest.raises(ValueError, match="gravitational_parameter"):
        force.compute_acceleration(0.0, np.zeros(6), None)


def test_indirect_term_3d_position():
    """三维天体位置下的加速度方向与大小正确。"""
    system = _FakeSystem()

    class _OffsetSystem(_FakeSystem):
        def get_body_position(self, body, t):
            return np.array([300000.0, 200000.0, 100000.0])

    force = IndirectTerm(body="MOON", mu=4902.800122)
    acc = force.compute_acceleration(0.0, np.zeros(6), _OffsetSystem())

    r = np.array([300000.0, 200000.0, 100000.0])
    r_norm = np.linalg.norm(r)
    expected = -4902.800122 / (r_norm**3) * r
    np.testing.assert_allclose(acc, expected)


def test_indirect_term_is_physical_model():
    """IndirectTerm 是 PhysicalModel 子类。"""
    from e2m2e.core.forces import PhysicalModel

    assert issubclass(IndirectTerm, PhysicalModel)


def test_indirect_term_body_normalized():
    """body 参数被 upper() 归一。"""
    force = IndirectTerm(body="moon")
    assert force.body == "MOON"


def test_indirect_term_properties():
    """property body/mu 正确暴露。"""
    force = IndirectTerm(body="MOON", mu=4902.800122)
    assert force.body == "MOON"
    assert force.mu == 4902.800122

    force_default = IndirectTerm(body="MOON")
    assert force_default.mu is None
