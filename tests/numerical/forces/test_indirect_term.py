"""IndirectTerm 测试。

Python 单点 ``compute_acceleration`` 已按 issue #378 删除；本文件验证
``to_rust_spec`` 序列化与 Rust 单点 ``indirect_term_acceleration`` 绑定。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import IndirectTerm
from e2m2e.integrators import indirect_term_acceleration

pytestmark = pytest.mark.force


class _FakeSystem:
    """最小 System 桩，提供 gravitational_parameter。"""

    coordinate_system = object()
    origin = "EARTH"

    def gravitational_parameter(self, body):
        if body.upper() == "EARTH":
            return 398600.4415
        if body.upper() == "MOON":
            return 4902.800122
        raise ValueError(f"unknown body {body}")


def test_indirect_term_with_explicit_mu_serializes_to_rust_spec():
    """显式传入 mu 时，to_rust_spec 携带该 mu。"""
    force = IndirectTerm(body="MOON", mu=4902.800122)
    spec = force.to_rust_spec(_FakeSystem())
    assert spec == ("indirect", "301", 4902.800122)


def test_indirect_term_falls_back_to_system_mu():
    """mu=None 时从 system.gravitational_parameter(body) 获取并写入 spec。"""
    force = IndirectTerm(body="MOON")
    spec = force.to_rust_spec(_FakeSystem())
    assert spec == ("indirect", "301", 4902.800122)


def test_indirect_term_earth_body_spec():
    """EARTH 体的 spec 与 MOON 体仅 body/mu 不同。"""
    force = IndirectTerm(body="EARTH", mu=398600.4415)
    spec = force.to_rust_spec(_FakeSystem())
    assert spec == ("indirect", "399", 398600.4415)


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
    force = IndirectTerm(body="MOON", mu=4902.800122)
    assert force.body == "MOON"
    assert force.mu == 4902.800122

    force_default = IndirectTerm(body="MOON")
    assert force_default.mu is None


def test_indirect_term_is_physical_model():
    """IndirectTerm 是 PhysicalModel 子类。"""
    from e2m2e.algorithm.forces import PhysicalModel

    assert issubclass(IndirectTerm, PhysicalModel)


@pytest.mark.spice
def test_indirect_term_rust_binding_matches_point_mass_formula(spice_manager, reference_epoch):
    """Rust ``indirect_term_acceleration`` 与 -mu·r/|r|³ 公式一致（SPICE 取位）。"""
    et = spice_manager.utc_to_et(reference_epoch)
    mu = 4902.800122
    acc = indirect_term_acceleration(et, "MOON", "EARTH", mu)

    r_moon = np.asarray(spice_manager.get_body_position("MOON", et, "J2000", "EARTH"), dtype=float)
    expected = -mu / np.linalg.norm(r_moon) ** 3 * r_moon
    np.testing.assert_allclose(acc, expected, rtol=1e-10)
