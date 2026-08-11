"""PointMassGravity 测试。

Python 单点 ``compute_acceleration`` 已按 issue #378 删除，加速度由 Rust
``propagate_compiled`` 承载；本文件验证 ``to_rust_spec`` 序列化与
``ForceModel`` 端到端传播的物理行为。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel, PointMassGravity

pytestmark = pytest.mark.force


class _FakeSystem:
    """最小 System 桩，提供 gravitational_parameter 与 coordinate_system。"""

    coordinate_system = object()
    origin = "EARTH"

    def gravitational_parameter(self, body):
        if body.upper() == "EARTH":
            return 398600.4415
        if body.upper() == "MOON":
            return 4902.800122
        raise ValueError(f"unknown body {body}")


def test_point_mass_gravity_with_explicit_mu_serializes_to_rust_spec():
    """显式传入 mu 时，to_rust_spec 携带该 mu。"""
    force = PointMassGravity(body="EARTH", mu=398600.4415)
    spec = force.to_rust_spec(_FakeSystem())
    assert spec == ("point_mass", 398600.4415)


def test_point_mass_gravity_falls_back_to_system_mu():
    """mu=None 时从 system.gravitational_parameter(body) 获取并写入 spec。"""
    force = PointMassGravity(body="EARTH")
    spec = force.to_rust_spec(_FakeSystem())
    assert spec == ("point_mass", 398600.4415)


def test_point_mass_gravity_moon_body():
    """换用 MOON 体，spec 使用月球 mu。"""
    force = PointMassGravity(body="MOON")
    spec = force.to_rust_spec(_FakeSystem())
    assert spec == ("point_mass", 4902.800122)


def test_point_mass_gravity_no_system_no_mu_raises():
    """mu=None 且 system 缺 gravitational_parameter 时抛 ValueError。"""
    force = PointMassGravity(body="EARTH")
    with pytest.raises(ValueError, match="gravitational_parameter"):
        force.to_rust_spec(None)


def test_point_mass_gravity_rust_propagation_matches_two_body_solution():
    """Rust compiled 传播点质量圆轨道，一个周期后回到初值。"""
    system = _FakeSystem()
    force = PointMassGravity(body="EARTH", mu=398600.4415)
    fm = ForceModel(system, forces=[force])

    r = 6778.0
    v = np.sqrt(398600.4415 / r)
    period = 2.0 * np.pi * np.sqrt(r**3 / 398600.4415)
    y0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    result = fm.propagate(y0, (0.0, period), t_eval=np.array([0.0, period]))

    final = result["states"][-1]
    np.testing.assert_allclose(final[:3], y0[:3], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(final[3:], y0[3:], rtol=1e-6, atol=1e-8)


def test_point_mass_gravity_is_physical_model():
    """PointMassGravity 是 PhysicalModel 子类。"""
    from e2m2e.algorithm.forces import PhysicalModel

    assert issubclass(PointMassGravity, PhysicalModel)
