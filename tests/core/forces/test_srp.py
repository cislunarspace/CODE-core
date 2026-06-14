"""SolarRadiationPressure 力模型测试。

纯函数路径（``_compute_srp_acceleration``）免 SPICE，与 ``test_drag.py`` 范式一致。
系统感知路径（``compute_acceleration``）由 ``test_srp_transform.py`` 覆盖。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.core.forces import PhysicalModel
from e2m2e.core.forces.srp import SolarRadiationPressure

# 1 AU（km），与 srp 模块内部常量一致。
_AU_KM = 149597870.691
# 太阳光压常数（N/m² @ 1 AU）。
_P_SRP_1AU = 4.56e-6


def test_srp_is_physical_model() -> None:
    """SolarRadiationPressure 是 PhysicalModel 的具体子类。"""
    srp = SolarRadiationPressure(area=10.0, mass=1000.0)
    assert isinstance(srp, PhysicalModel)


def test_srp_magnitude_at_1au_matches_canonical_formula() -> None:
    """1 AU 处光压加速度量级 = P·Cr·A/m（验收 1）。

    cannonball 模型：a = P · (1AU/r)² · Cr·A/m。1 AU 处 (1AU/r)²=1。
    """
    cr, area, mass = 1.5, 10.0, 1000.0
    srp = SolarRadiationPressure(area=area, mass=mass, cr=cr)

    sun_to_sc = np.array([_AU_KM, 0.0, 0.0])
    acc = srp._compute_srp_acceleration(sun_to_sc, flux_factor=1.0)

    expected_si = _P_SRP_1AU * cr * area / mass  # m/s²
    expected_km = expected_si / 1000.0  # km/s²
    np.testing.assert_allclose(np.linalg.norm(acc), expected_km, rtol=1e-10)


def test_srp_direction_points_away_from_sun() -> None:
    """加速度沿 Sun→SC 方向（远离太阳），跟随向量方向而非硬编码轴。"""
    srp = SolarRadiationPressure(area=10.0, mass=1000.0)

    # Sun→SC 沿 +y，加速度应纯 +y
    sun_to_sc = np.array([0.0, _AU_KM, 0.0])
    acc = srp._compute_srp_acceleration(sun_to_sc, flux_factor=1.0)

    assert acc[1] > 0.0
    np.testing.assert_allclose(acc[0], 0.0, atol=1e-30)
    np.testing.assert_allclose(acc[2], 0.0, atol=1e-30)


def test_srp_inverse_square_scaling() -> None:
    """2 AU 处量级 = 1 AU 处的 1/4（1/r² 标度）。"""
    srp = SolarRadiationPressure(area=10.0, mass=1000.0)

    acc_1au = srp._compute_srp_acceleration(np.array([_AU_KM, 0.0, 0.0]), 1.0)
    acc_2au = srp._compute_srp_acceleration(np.array([2.0 * _AU_KM, 0.0, 0.0]), 1.0)

    ratio = np.linalg.norm(acc_2au) / np.linalg.norm(acc_1au)
    np.testing.assert_allclose(ratio, 0.25, rtol=1e-12)


def test_srp_scales_linearly_with_cr() -> None:
    """Cr 翻倍则加速度量级翻倍。"""
    srp1 = SolarRadiationPressure(area=10.0, mass=1000.0, cr=1.0)
    srp2 = SolarRadiationPressure(area=10.0, mass=1000.0, cr=2.0)

    sun_to_sc = np.array([_AU_KM, 0.0, 0.0])
    a1 = np.linalg.norm(srp1._compute_srp_acceleration(sun_to_sc, 1.0))
    a2 = np.linalg.norm(srp2._compute_srp_acceleration(sun_to_sc, 1.0))

    np.testing.assert_allclose(a2 / a1, 2.0, rtol=1e-12)


def test_srp_scales_with_flux_factor() -> None:
    """flux_factor=0.5 给半量；flux_factor=0 给零向量（本影）。"""
    srp = SolarRadiationPressure(area=10.0, mass=1000.0)
    sun_to_sc = np.array([_AU_KM, 0.0, 0.0])

    full = np.linalg.norm(srp._compute_srp_acceleration(sun_to_sc, 1.0))
    half = np.linalg.norm(srp._compute_srp_acceleration(sun_to_sc, 0.5))
    dark = srp._compute_srp_acceleration(sun_to_sc, 0.0)

    np.testing.assert_allclose(half / full, 0.5, rtol=1e-12)
    np.testing.assert_array_equal(dark, np.zeros(3))


def test_srp_rejects_nonpositive_area() -> None:
    """截面积必须为正。"""
    with pytest.raises(ValueError, match="area"):
        SolarRadiationPressure(area=0.0, mass=1000.0)


def test_srp_rejects_nonpositive_mass() -> None:
    """质量必须为正。"""
    with pytest.raises(ValueError, match="mass"):
        SolarRadiationPressure(area=10.0, mass=-5.0)
