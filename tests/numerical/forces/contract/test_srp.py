"""SolarRadiationPressure 力模型测试。

Python 单点 ``compute_acceleration`` 与 ``_compute_srp_acceleration`` 已删除；
SRP 物理行为由 Rust ``propagate_compiled`` 与
``srp_acceleration`` 绑定承载。本文件保留构造校验与 ``to_rust_spec``
序列化契约。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.forces import PhysicalModel
from e2m2e.algorithm.forces.srp import SolarRadiationPressure

pytestmark = pytest.mark.force


def test_srp_is_physical_model() -> None:
    """SolarRadiationPressure 是 PhysicalModel 的具体子类。"""
    srp = SolarRadiationPressure(area=10.0, mass=1000.0)
    assert isinstance(srp, PhysicalModel)


def test_srp_rejects_nonpositive_area() -> None:
    """截面积必须为正。"""
    with pytest.raises(ValueError, match="area"):
        SolarRadiationPressure(area=0.0, mass=1000.0)


def test_srp_rejects_nonpositive_mass() -> None:
    """质量必须为正。"""
    with pytest.raises(ValueError, match="mass"):
        SolarRadiationPressure(area=10.0, mass=-5.0)


def test_srp_to_rust_spec_without_shadow() -> None:
    """无阴影时 to_rust_spec 返回 ("srp", area, mass, cr, [])。"""
    srp = SolarRadiationPressure(area=10.0, mass=1000.0, cr=1.5)
    spec = srp.to_rust_spec(None)
    assert spec == ("srp", 10.0, 1000.0, 1.5, [])


def test_srp_to_rust_spec_with_shadow() -> None:
    """含阴影时 to_rust_spec 把 shadow bodies 带进元组。"""
    from e2m2e.algorithm.forces.shadow import ConicalShadowModel

    shadow = ConicalShadowModel(bodies=["EARTH", "MOON"])
    srp = SolarRadiationPressure(area=5.0, mass=500.0, cr=1.2, shadow=shadow)
    spec = srp.to_rust_spec(None)
    assert spec == ("srp", 5.0, 500.0, 1.2, ["EARTH", "MOON"])


def test_srp_stores_injected_shadow() -> None:
    """SRP 持有注入的阴影模型实例。"""
    from e2m2e.algorithm.forces.shadow import ConicalShadowModel

    shadow = ConicalShadowModel(bodies=["EARTH", "MOON"])
    srp = SolarRadiationPressure(area=10.0, mass=1000.0, shadow=shadow)
    assert srp.shadow is shadow


def test_variable_mass_srp_is_physical_model() -> None:
    """VariableMassSolarRadiationPressure 是 PhysicalModel 的具体子类。"""
    from e2m2e.algorithm.forces.srp import VariableMassSolarRadiationPressure

    srp = VariableMassSolarRadiationPressure(area=20.0)
    assert isinstance(srp, PhysicalModel)


def test_variable_mass_srp_rejects_nonpositive_area() -> None:
    """截面积必须为正。"""
    from e2m2e.algorithm.forces.srp import VariableMassSolarRadiationPressure

    with pytest.raises(ValueError, match="area"):
        VariableMassSolarRadiationPressure(area=0.0)


def test_variable_mass_srp_to_rust_spec_without_shadow() -> None:
    """无阴影时 to_rust_spec 返回 ("srp_variable_mass", area, cr, [])，不带质量。"""
    from e2m2e.algorithm.forces.srp import VariableMassSolarRadiationPressure

    srp = VariableMassSolarRadiationPressure(area=20.0, cr=1.3)
    spec = srp.to_rust_spec(None)
    assert spec == ("srp_variable_mass", 20.0, 1.3, [])


def test_variable_mass_srp_to_rust_spec_with_shadow() -> None:
    """含阴影时 to_rust_spec 把 shadow bodies 带进元组。"""
    from e2m2e.algorithm.forces.shadow import ConicalShadowModel
    from e2m2e.algorithm.forces.srp import VariableMassSolarRadiationPressure

    shadow = ConicalShadowModel(bodies=["EARTH", "MOON"])
    srp = VariableMassSolarRadiationPressure(area=20.0, cr=1.3, shadow=shadow)
    spec = srp.to_rust_spec(None)
    assert spec == ("srp_variable_mass", 20.0, 1.3, ["EARTH", "MOON"])
