"""IndirectTerm 定义与序列化契约。

验证构造校验、body 归一化、``to_rust_spec`` 序列化与 mu fallback；Rust 单点
``indirect_term_acceleration`` 的物理对照在 ``physics/test_indirect_term.py``。
"""

import pytest

from e2m2e.algorithm.forces import IndirectTerm
from tests.numerical.forces.conftest import EARTH_MU, MOON_MU, FakeSystem

pytestmark = pytest.mark.force


def test_indirect_term_with_explicit_mu_serializes_to_rust_spec():
    """显式传入 mu 时，to_rust_spec 携带该 mu。"""
    force = IndirectTerm(body="MOON", mu=MOON_MU)
    spec = force.to_rust_spec(FakeSystem())
    assert spec == ("indirect", "301", MOON_MU)


def test_indirect_term_falls_back_to_system_mu():
    """mu=None 时从 system.gravitational_parameter(body) 获取并写入 spec。"""
    force = IndirectTerm(body="MOON")
    spec = force.to_rust_spec(FakeSystem())
    assert spec == ("indirect", "301", MOON_MU)


def test_indirect_term_earth_body_spec():
    """EARTH 体的 spec 与 MOON 体仅 body/mu 不同。"""
    force = IndirectTerm(body="EARTH", mu=EARTH_MU)
    spec = force.to_rust_spec(FakeSystem())
    assert spec == ("indirect", "399", EARTH_MU)


def test_indirect_term_no_system_no_mu_raises():
    """mu=None 且 system 缺 gravitational_parameter 时抛 ValueError。"""
    force = IndirectTerm(body="MOON")
    with pytest.raises(ValueError, match="gravitational_parameter"):
        force.to_rust_spec(None)


def test_indirect_term_body_normalized():
    """body 参数被 upper() 归一。"""
    force = IndirectTerm(body="moon")
    assert force.body == "MOON"


def test_indirect_term_properties():
    """property body/mu 正确暴露。"""
    force = IndirectTerm(body="MOON", mu=MOON_MU)
    assert force.body == "MOON"
    assert force.mu == MOON_MU

    force_default = IndirectTerm(body="MOON")
    assert force_default.mu is None


def test_indirect_term_is_physical_model():
    """IndirectTerm 是 PhysicalModel 子类。"""
    from e2m2e.algorithm.forces import PhysicalModel

    assert issubclass(IndirectTerm, PhysicalModel)
