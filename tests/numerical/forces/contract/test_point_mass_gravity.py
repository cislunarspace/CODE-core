"""PointMassGravity 定义与序列化契约。

验证 ``to_rust_spec`` 序列化（显式 mu / system fallback / 无 system 报错）
与 PhysicalModel 子类关系；物理行为见 ``physics/test_point_mass_gravity.py``。
"""

import pytest

from e2m2e.algorithm.forces import PointMassGravity
from tests.numerical.forces.conftest import EARTH_MU, MOON_MU, FakeSystem

pytestmark = pytest.mark.force


def test_point_mass_gravity_with_explicit_mu_serializes_to_rust_spec():
    """显式传入 mu 时，to_rust_spec 携带该 mu。"""
    force = PointMassGravity(body="EARTH", mu=EARTH_MU)
    spec = force.to_rust_spec(FakeSystem())
    assert spec == ("point_mass", EARTH_MU)


def test_point_mass_gravity_falls_back_to_system_mu():
    """mu=None 时从 system.gravitational_parameter(body) 获取并写入 spec。"""
    force = PointMassGravity(body="EARTH")
    spec = force.to_rust_spec(FakeSystem())
    assert spec == ("point_mass", EARTH_MU)


def test_point_mass_gravity_moon_body():
    """换用 MOON 体，spec 使用月球 mu。"""
    force = PointMassGravity(body="MOON")
    spec = force.to_rust_spec(FakeSystem())
    assert spec == ("point_mass", MOON_MU)


def test_point_mass_gravity_no_system_no_mu_raises():
    """mu=None 且 system 缺 gravitational_parameter 时抛 ValueError。"""
    force = PointMassGravity(body="EARTH")
    with pytest.raises(ValueError, match="gravitational_parameter"):
        force.to_rust_spec(None)


def test_point_mass_gravity_is_physical_model():
    """PointMassGravity 是 PhysicalModel 子类。"""
    from e2m2e.algorithm.forces import PhysicalModel

    assert issubclass(PointMassGravity, PhysicalModel)
