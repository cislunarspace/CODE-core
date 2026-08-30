"""Primer §5 解析尺度的理论黄金值测试（Rosengren et al. 2026）。

黄金值来源：论文 Table 1/2 与 §5 正文数值，另经独立数值复算核验
（SPICE GM + Simon 1994 月根数 + GGM02 J2；au 取 IAU 精确值）。
复现陷阱（ADR 0041）在对应用例内嵌断言。
"""

from __future__ import annotations

import math

import pytest

from e2m2e.algorithm.spatiography.constants import PRIMER_DEFAULTS
from e2m2e.algorithm.spatiography.scales import (
    activity_surface_moon,
    battin_soi_earth,
    battin_soi_moon,
    characteristic_rate_j2,
    characteristic_rate_lunar_exterior,
    characteristic_rate_solar_exterior,
    chebotarev_radius_earth,
    chebotarev_radius_moon,
    geo_radius_km,
    hill_radius_earth,
    hill_radius_moon,
    laplace_radius_geolunar,
    laplace_radius_selenocentric,
    soi_laplace_earth,
    soi_laplace_moon,
    tidal_parity_radius,
    tisserand_parameter,
)

pytestmark = pytest.mark.theory

_MOON_RADIUS_KM = 1737.4  # IAU2015（论文表值反推一致）


def test_laplace_radius_geolunar_golden_value():
    """式(98/99)：r_L = 48812.40 km = 7.6531 R+ = 0.12732 a☾。"""
    r_l = laplace_radius_geolunar()
    assert r_l == pytest.approx(48812.40, rel=1e-4)
    assert r_l / PRIMER_DEFAULTS.earth_ref_radius_km == pytest.approx(7.6531, rel=1e-3)
    assert r_l / PRIMER_DEFAULTS.moon_a_km == pytest.approx(0.12732, rel=1e-3)


def test_laplace_radius_satisfies_the_characteristic_rate_balance():
    """式(98) 与式(95/96/97) 自洽：a = r_L 处 omega^+ = omega^☾ + omega^☉。"""
    r_l = laplace_radius_geolunar()
    balance = (
        characteristic_rate_j2(r_l)
        - characteristic_rate_lunar_exterior(r_l)
        - characteristic_rate_solar_exterior(r_l)
    )
    assert balance == pytest.approx(0.0, rel=1e-9, abs=1e-18)


def test_laplace_radius_selenocentric_golden_value():
    """式(124/125)：rho_L = 3846 km = 2.21 R☾（高约 2109 km）。"""
    rho_l = laplace_radius_selenocentric()
    assert rho_l == pytest.approx(3846.0, rel=1e-3)
    assert rho_l / _MOON_RADIUS_KM == pytest.approx(2.21, rel=1e-2)


def test_moon_hill_radius_requires_the_approximate_form():
    """陷阱⑤：论文 61364 km（35.32 R☾）只由近似式 a☾(μ☾/3μ⊕)^{1/3} 复现；
    完整式（分母 3(μ⊕+μ☾)）给 61114 km，差约 -250 km，不得混用。"""
    rho_h = hill_radius_moon()
    assert rho_h == pytest.approx(61364.0, rel=1e-4)
    assert rho_h / _MOON_RADIUS_KM == pytest.approx(35.32, rel=1e-3)
    c = PRIMER_DEFAULTS
    complete_form = c.moon_a_km * (c.moon_gm / (3.0 * (c.earth_gm + c.moon_gm))) ** (1.0 / 3.0)
    assert complete_form == pytest.approx(61114.0, rel=1e-3)
    assert complete_form < rho_h - 200.0


def test_earth_hill_radius_golden_value():
    r_h = hill_radius_earth()
    assert r_h / PRIMER_DEFAULTS.moon_a_km == pytest.approx(3.9034, rel=1e-4)
    assert r_h == pytest.approx(1.4966e6, rel=1e-3)


def test_moon_soi_and_activity_surface():
    """式(115/116)：球形代理 66010 km；活动面在方照 psi=pi/2 处取该最大值。"""
    soi = soi_laplace_moon()
    assert soi == pytest.approx(66010.0, rel=1e-4)
    assert soi / _MOON_RADIUS_KM == pytest.approx(37.99, rel=1e-3)
    assert activity_surface_moon(math.pi / 2.0) == pytest.approx(soi, rel=1e-9)
    # psi=0（背地）压缩、psi=pi（朝地）更小——活动面关于 psi=pi/2 对称。
    assert activity_surface_moon(0.0) < soi
    assert activity_surface_moon(math.pi) == pytest.approx(activity_surface_moon(0.0), rel=1e-9)


def test_moon_chebotarev_radius_paper_rounding_note():
    """式(117)：复算 42520.9 km；论文 km 值 42499 差 0.05%（内部舍入）。"""
    rho_ch = chebotarev_radius_moon()
    assert rho_ch == pytest.approx(42520.9, rel=1e-4)
    assert rho_ch / _MOON_RADIUS_KM == pytest.approx(24.47, rel=1e-3)
    assert rho_ch == pytest.approx(42499.0, rel=6e-4)


def test_battin_moon_soi_anti_earthward_zero_point():
    """陷阱④：psi 零点取反地方向——psi=0 背地 64201 km、psi=pi 朝地 52009 km。

    论文正文行内定义（月卫线与月地线夹角）与其数值矛盾，按数值复现约定实现。
    另注：曲线全局最大值并不在 psi=0 处（约 66.4 Mm、psi~78°），论文仅引用
    轴向值。
    """
    assert battin_soi_moon(0.0) == pytest.approx(64201.0, rel=1e-4)
    assert battin_soi_moon(math.pi) == pytest.approx(52009.0, rel=1e-4)
    radii = [battin_soi_moon(psi) for psi in [i * math.pi / 180.0 for i in range(0, 360, 2)]]
    assert max(radii) == pytest.approx(66389.0, rel=5e-3)


def test_earth_scales_golden_values_in_lunar_units():
    assert soi_laplace_earth() / PRIMER_DEFAULTS.moon_a_km == pytest.approx(2.4117, rel=1e-3)
    assert chebotarev_radius_earth() / PRIMER_DEFAULTS.moon_a_km == pytest.approx(0.6762, rel=1e-3)
    # 日地质量比下 Battin 变形微弱：方照回到球形 SOI（式 122 注记）。
    assert battin_soi_earth(math.pi / 2.0) == pytest.approx(soi_laplace_earth(), rel=1e-3)


def test_tidal_parity_radius_golden_value():
    """式(127/128)：a_TP = 447948 km = 1.1684 a☾，T = 34.53 天（表值 34.6 为
    按舍入 1.17 重算值，容差按陷阱注记放宽）。"""
    a_tp = tidal_parity_radius()
    assert a_tp == pytest.approx(447948.0, rel=1e-4)
    assert a_tp / PRIMER_DEFAULTS.moon_a_km == pytest.approx(1.1684, rel=1e-3)
    t_days = PRIMER_DEFAULTS.moon_period_days * (a_tp / PRIMER_DEFAULTS.moon_a_km) ** 1.5
    assert t_days == pytest.approx(34.53, abs=0.1)


def test_moon_period_is_derived_not_hardcoded():
    """陷阱①：T☾ 必须解析派生（27.34460 天），不得硬编码 27.346。"""
    t_moon = PRIMER_DEFAULTS.moon_period_days
    assert t_moon == pytest.approx(27.34460, rel=1e-5)
    assert abs(t_moon - 27.346) > 1.0e-3


def test_tisserand_circular_coplanar_reference_value():
    """式(140)：共面 a=a☾ 圆轨道 T☾=3——参考值而非 gateway 阈值。"""
    assert tisserand_parameter(PRIMER_DEFAULTS.moon_a_km, 0.0, 0.0) == pytest.approx(3.0, rel=1e-9)


def test_geo_radius_matches_standard_value():
    """GEO 半径按恒星日派生 ~42164 km（仅作走廊参考线，非分区判据）。"""
    assert geo_radius_km() == pytest.approx(42164.0, rel=1e-3)
