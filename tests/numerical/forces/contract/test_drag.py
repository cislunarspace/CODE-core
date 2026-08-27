"""DragModel 大气阻力力模型测试。

Python 单点 ``compute_acceleration`` 已删除；阻力物理行为由
Rust ``drag_accel_in_body_fixed`` 单元测试与 Rust ``propagate_compiled``
端到端传播覆盖。本文件保留构造校验、弹道系数与 ``to_rust_spec`` 序列化契约。
"""

import types

import pytest

from e2m2e.algorithm.forces import PhysicalModel
from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere
from e2m2e.algorithm.forces.drag import DragModel

pytestmark = pytest.mark.force


def _system_with_spice():
    """最小 system 桩：to_rust_spec 只检查 getattr(system, 'spice') 非 None。"""
    return types.SimpleNamespace(spice=object())


def test_drag_model_is_physical_model():
    """DragModel 是 PhysicalModel 的具体子类。"""
    atm = ExponentialAtmosphere()
    drag = DragModel(atmosphere=atm, area=10.0, mass=1000.0)
    assert isinstance(drag, PhysicalModel)


def test_drag_model_rejects_nonpositive_area():
    """截面积必须为正。"""
    atm = ExponentialAtmosphere()
    with pytest.raises(ValueError, match="area"):
        DragModel(atmosphere=atm, area=0.0, mass=1000.0)


def test_drag_model_rejects_nonpositive_mass():
    """质量必须为正。"""
    atm = ExponentialAtmosphere()
    with pytest.raises(ValueError, match="mass"):
        DragModel(atmosphere=atm, area=10.0, mass=-5.0)


def test_drag_model_rejects_nonpositive_cd():
    """阻力系数必须为正。"""
    atm = ExponentialAtmosphere()
    with pytest.raises(ValueError, match="cd"):
        DragModel(atmosphere=atm, area=10.0, mass=1000.0, cd=0.0)


def test_drag_ballistic_coefficient():
    """弹道系数 = Cd·A/m。"""
    atm = ExponentialAtmosphere()
    drag = DragModel(atmosphere=atm, area=10.0, mass=1000.0, cd=2.2)
    assert drag.ballistic_coefficient == pytest.approx(2.2 * 10.0 / 1000.0)


def test_to_rust_spec_carries_f107_ap():
    """to_rust_spec 必须把大气模型的 f107/ap 带进元组。

    Rust 路径从元组取这两项传给 density；若缺项，Rust 会用硬编码默认值，
    与用户配置静默分歧（f107=200 时 ~17% 偏移）。
    """
    system = _system_with_spice()

    # 默认大气
    drag_default = DragModel(atmosphere=ExponentialAtmosphere(), area=10.0, mass=1000.0, cd=2.2)
    spec_default = drag_default.to_rust_spec(system)
    assert spec_default is not None
    assert spec_default[0] == "drag"
    assert spec_default[1:5] == (10.0, 1000.0, 2.2, "J2000")
    # 第 5、6 位是 f107、ap
    assert spec_default[5] == pytest.approx(150.0)
    assert spec_default[6] == pytest.approx(15.0)

    # 非默认大气：f107/ap 必须随大气模型变化
    drag_hot = DragModel(
        atmosphere=ExponentialAtmosphere(f107=200.0, ap=50.0),
        area=10.0,
        mass=1000.0,
        cd=2.2,
    )
    spec_hot = drag_hot.to_rust_spec(system)
    assert spec_hot is not None
    assert spec_hot[5] == pytest.approx(200.0)
    assert spec_hot[6] == pytest.approx(50.0)


def test_to_rust_spec_none_without_spice():
    """system 无 spice 属性时 to_rust_spec 返回 None，由 ForceModel 显式报错。"""
    drag = DragModel(atmosphere=ExponentialAtmosphere(), area=10.0, mass=1000.0)
    assert drag.to_rust_spec(types.SimpleNamespace()) is None


def test_to_rust_spec_none_for_non_earth_body():
    """中心天体非 EARTH 时 to_rust_spec 返回 None（仅支持地球阻力）。"""
    drag = DragModel(atmosphere=ExponentialAtmosphere(), area=10.0, mass=1000.0, body="MOON")
    assert drag.to_rust_spec(_system_with_spice()) is None
