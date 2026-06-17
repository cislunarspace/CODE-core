"""DragModel 大气阻力力模型测试。

覆盖 PhysicalModel 子类关系、加速度方向、量级公式与边界。
"""

import numpy as np
import pytest

from e2m2e.core.atmosphere import ExponentialAtmosphere
from e2m2e.core.forces import PhysicalModel
from e2m2e.core.forces.drag import DragModel

_EARTH_R_KM = 6378.137
_KM_TO_M = 1000.0


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
