"""ExponentialAtmosphere 大气密度模型测试。

覆盖 USSA76 参考值、高度单调性、F10.7/Ap 修正与边界钳制。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere

pytestmark = pytest.mark.force


# USSA76 指数模型基准密度（kg/m³），用于验证误差 < 20%。
# 这些是 US Standard Atmosphere 1976 分段指数拟合的标准断点值。
USS76_REFERENCE_DENSITY = {
    0: 1.225e0,
    100: 5.604e-7,
    200: 2.541e-10,
    300: 1.916e-11,
    400: 2.803e-12,
    500: 5.215e-13,
    700: 3.381e-14,
}


def test_density_returns_positive_float():
    """density 返回正浮点数（海平面附近密度 > 0）。"""
    atm = ExponentialAtmosphere()
    rho = atm.density(10.0)  # 10 km altitude
    assert isinstance(rho, float)
    assert rho > 0.0


def test_density_at_sea_level_matches_uss76():
    """海平面密度 ≈ 1.225 kg/m³（USSA76 标准值）。"""
    atm = ExponentialAtmosphere()
    rho = atm.density(0.0)
    np.testing.assert_allclose(rho, 1.225, rtol=1e-6)


@pytest.mark.parametrize("altitude_km", list(USS76_REFERENCE_DENSITY.keys()))
def test_density_within_20_percent_of_uss76(altitude_km):
    """多个高度的密度与 USSA76 参考值误差 < 20%。"""
    atm = ExponentialAtmosphere()
    rho = atm.density(altitude_km)
    expected = USS76_REFERENCE_DENSITY[altitude_km]
    relative_error = abs(rho - expected) / expected
    assert relative_error < 0.20, (
        f"altitude={altitude_km} km: rho={rho:.4e}, expected={expected:.4e}, "
        f"error={relative_error:.1%}"
    )


def test_density_monotonically_decreases_with_altitude():
    """密度随高度单调递减。"""
    atm = ExponentialAtmosphere()
    altitudes = np.linspace(0, 950, 50)
    densities = [atm.density(h) for h in altitudes]
    diffs = np.diff(densities)
    assert np.all(diffs < 0), "密度应随高度单调递减"


def test_higher_f107_gives_higher_density():
    """更高的 F10.7 太阳通量 → 更高的大气密度。"""
    atm_low = ExponentialAtmosphere(f107=100)
    atm_high = ExponentialAtmosphere(f107=200)
    rho_low = atm_low.density(400.0)
    rho_high = atm_high.density(400.0)
    assert rho_high > rho_low


def test_higher_ap_gives_higher_density():
    """更高的 Ap 地磁指数 → 更高的大气密度。"""
    atm_low = ExponentialAtmosphere(ap=5)
    atm_high = ExponentialAtmosphere(ap=50)
    rho_low = atm_low.density(400.0)
    rho_high = atm_high.density(400.0)
    assert rho_high > rho_low


def test_default_f107_and_ap_give_baseline_density():
    """默认 F10.7=150、Ap=15 时，密度修正因子为 1（等于参考密度）。"""
    atm = ExponentialAtmosphere()  # defaults
    rho = atm.density(400.0)
    np.testing.assert_allclose(rho, USS76_REFERENCE_DENSITY[400], rtol=1e-9)


def test_density_above_ceiling_is_zero():
    """高度超过模型上限（1000 km）时密度为 0。"""
    atm = ExponentialAtmosphere()
    assert atm.density(1000.0) == 0.0
    assert atm.density(1500.0) == 0.0


def test_density_below_zero_clamped_to_surface():
    """高度低于 0 km 时钳到 0 km，返回地表密度而非负高度爆炸。"""
    atm = ExponentialAtmosphere()
    rho = atm.density(-10.0)
    assert rho > 0.0
    # 钳到 0 km 时密度应等于海平面密度
    np.testing.assert_allclose(rho, atm.density(0.0), rtol=1e-12)
