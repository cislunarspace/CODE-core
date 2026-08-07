"""DragModel 大气阻力力模型测试。

覆盖 PhysicalModel 子类关系、加速度方向、量级公式与边界。
"""

import types

import numpy as np
import pytest

from e2m2e.algorithm.forces import PhysicalModel
from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere
from e2m2e.algorithm.forces.drag import DragModel
from e2m2e.data.templates.systems import KM_TO_M as _KM_TO_M
from e2m2e.data.templates.systems import R_EARTH as _EARTH_R_KM


def _system_with_spice():
    """最小 system 桩：to_rust_spec 只检查 getattr(system, 'spice') 非 None。"""
    return types.SimpleNamespace(spice=object())


def test_drag_model_is_physical_model():
    """DragModel 是 PhysicalModel 的具体子类。"""
    atm = ExponentialAtmosphere()
    drag = DragModel(atmosphere=atm, area=10.0, mass=1000.0)
    assert isinstance(drag, PhysicalModel)


def test_compute_acceleration_returns_three_vector():
    """compute_acceleration 返回形状 (3,) 的加速度向量。

    system=None 时假设状态已在 ITRF 中（与 GravityField 隔离测试模式一致）。
    """
    atm = ExponentialAtmosphere()
    drag = DragModel(atmosphere=atm, area=10.0, mass=1000.0)

    # 400 km 高度，ITRF 中位置 [km]、速度 [km/s]
    state = np.array([6778.0, 0.0, 0.0, 0.0, 7.7, 0.0])
    acc = drag.compute_acceleration(0.0, state, None)

    assert isinstance(acc, np.ndarray)
    assert acc.shape == (3,)


def test_drag_acceleration_opposes_velocity():
    """阻力加速度方向与速度方向相反。"""
    atm = ExponentialAtmosphere()
    drag = DragModel(atmosphere=atm, area=10.0, mass=1000.0)

    # 速度在 +y 方向，阻力应在 -y 方向
    state = np.array([6778.0, 0.0, 0.0, 0.0, 7.7, 0.0])
    acc = drag.compute_acceleration(0.0, state, None)

    assert acc[1] < 0.0, "阻力 y 分量应为负（反速度方向）"
    # x、z 分量应可忽略（速度纯 y 方向）
    assert abs(acc[0]) < abs(acc[1]) * 1e-12
    assert abs(acc[2]) < abs(acc[1]) * 1e-12


def test_drag_magnitude_matches_formula():
    """阻力加速度量级与解析公式 a = 0.5·ρ·BC·v² 一致（验证单位转换正确）。"""
    atm = ExponentialAtmosphere()
    cd, area, mass = 2.2, 10.0, 1000.0
    drag = DragModel(atmosphere=atm, area=area, mass=mass, cd=cd)

    state = np.array([6778.0, 0.0, 0.0, 0.0, 7.7, 0.0])
    acc = drag.compute_acceleration(0.0, state, None)

    # 手工计算（SI）
    altitude = 6778.0 - _EARTH_R_KM
    rho = atm.density(altitude)  # kg/m³
    v = 7.7 * _KM_TO_M  # m/s
    bc = cd * area / mass  # m²/kg
    a_expected_si = 0.5 * rho * bc * v**2  # m/s²
    a_expected_km = a_expected_si / _KM_TO_M  # km/s²

    np.testing.assert_allclose(np.linalg.norm(acc), a_expected_km, rtol=1e-10)


def test_drag_above_ceiling_is_zero():
    """高度超过大气模型上限（1000 km）时阻力为零。"""
    atm = ExponentialAtmosphere()
    drag = DragModel(atmosphere=atm, area=10.0, mass=1000.0)

    # ~1200 km 高度（超过 1000 km 上限）
    state = np.array([7578.0, 0.0, 0.0, 0.0, 6.0, 0.0])
    acc = drag.compute_acceleration(0.0, state, None)
    np.testing.assert_array_equal(acc, np.zeros(3))


def test_drag_zero_velocity_is_zero():
    """相对速度为零时阻力为零。"""
    atm = ExponentialAtmosphere()
    drag = DragModel(atmosphere=atm, area=10.0, mass=1000.0)

    state = np.array([6778.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    acc = drag.compute_acceleration(0.0, state, None)
    np.testing.assert_array_equal(acc, np.zeros(3))


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


def test_to_rust_spec_carries_f107_ap():
    """#315：to_rust_spec 必须把大气模型的 f107/ap 带进元组。

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
    """system 无 spice 属性时 to_rust_spec 返回 None，回退 Python 路径。"""
    drag = DragModel(atmosphere=ExponentialAtmosphere(), area=10.0, mass=1000.0)
    assert drag.to_rust_spec(types.SimpleNamespace()) is None
