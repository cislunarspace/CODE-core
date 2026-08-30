"""长期共振 loci 与 vZLK 相图测试（论文 §4.2–§4.3 式 64–78）。

验收口径（issue #578）：
- loci 水平线 5cos²I = 1−e²（式 76）与式 78 曲线的两个极限
  （1/5 与 (1−e²)/5，式 79/81）数值验证；
- 黄金值可对式号逐一溯源；中文 docstring 全角括号坑不适用于测试断言。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from e2m2e.algorithm.spatiography import secular as s
from e2m2e.algorithm.spatiography.constants import PRIMER_DEFAULTS

pytestmark = pytest.mark.theory

C = PRIMER_DEFAULTS


class TestSecularPrefactors:
    def test_eq47_and_eq56_prefactor_scaling(self):
        """式 47：ω_ext ∝ a^{3/2}；式 56：ω_int ∝ a^{-7/2}。"""
        a1, a2 = 2.0e5, 3.0e5
        ratio_ext = s.secular_prefactor_ext_sun(a2) / s.secular_prefactor_ext_sun(a1)
        assert ratio_ext == pytest.approx((a2 / a1) ** 1.5, rel=1e-12)
        ratio_int = s.secular_prefactor_int_moon(a2) / s.secular_prefactor_int_moon(a1)
        assert ratio_int == pytest.approx((a2 / a1) ** -3.5, rel=1e-12)

    def test_branches_nearly_continuous_at_lunar_distance(self):
        """外支/内支前置频率在 a = a☾ 处衔接（Hansen 排布互换的近似恒等）。"""
        w_ext = s.secular_prefactor_ext_moon(C.moon_a_km)
        w_int = s.secular_prefactor_int_moon(C.moon_a_km)
        assert w_int / w_ext == pytest.approx(1.0, rel=1e-3)

    def test_kaula_k_factor_differs_from_laplace_eq96(self):
        """式 48 的 K = 1 − (3/2)sin²I☾（0.9879）≠ 式 96 的 1 − sin²I☾/2。"""
        from e2m2e.algorithm.spatiography.scales import characteristic_rate_lunar_exterior

        inc = math.radians(C.moon_inc_deg)
        k_48 = 1.0 - 1.5 * math.sin(inc) ** 2
        ratio = characteristic_rate_lunar_exterior(C.moon_a_km) / (
            0.75
            * (C.moon_gm / math.sqrt(C.earth_gm))
            * C.moon_a_km**1.5
            / (C.moon_a_km**3 * (1.0 - C.moon_ecc**2) ** 1.5)
        )
        assert ratio == pytest.approx(1.0 - 0.5 * math.sin(inc) ** 2, rel=1e-12)
        assert abs(k_48 - ratio) > 0.005


class TestApsidalStationaryLoci:
    def test_eq76_horizontal_line_zeros_combined_rate(self):
        """式 76：5cos²I = 1−e² 处合并拱线率（式 75）为零（任意 a）。"""
        for ecc in (0.0, 0.3, 0.6):
            inc = math.acos(math.sqrt((1.0 - ecc**2) / 5.0))
            for a_km in (6.0e4, 1.5e5, 3.0e5):
                assert s.apsidal_rate_cislunar(a_km, ecc, inc) == pytest.approx(0.0, abs=1e-21), (
                    ecc,
                    a_km,
                )

    def test_eq78_closed_form_is_root_of_eq77(self):
        a_km = 2.0 * C.moon_a_km
        root = brentq(lambda i: s.apsidal_rate_translunar(a_km, 0.3, i), 0.01, math.pi / 2 - 1e-6)
        closed = s.apsidal_stationary_inclination_translunar(a_km, 0.3)
        assert closed == pytest.approx(root, rel=1e-10)

    def test_eq78_limits_interior_lunar_and_solar_exterior(self):
        """式 79/81 两个极限：月内主导 cos²I → 1/5；日外主导 → (1−e²)/5。"""
        ecc = 0.3
        near = s.apsidal_stationary_inclination_translunar(1.0001 * C.moon_a_km, ecc)
        far = s.apsidal_stationary_inclination_translunar(30.0 * C.moon_a_km, ecc)
        # 月内主导极限：63.4°/116.6°（式 80）；贴月处日外项仅 ~0.05%，
        # cos²I 已进入 (0.19, 0.2) 区间单调趋近 1/5。
        assert 63.4 < math.degrees(near) < 64.2
        assert 0.19 < math.cos(near) ** 2 < 0.2
        # 日外主导极限：(1−e²)/5（式 81）。
        assert math.cos(far) ** 2 == pytest.approx((1.0 - ecc**2) / 5.0, rel=1e-6)

    def test_loci_curves_carry_formula_ids(self):
        curves = s.secular_loci_curves(e_slices=(0.0, 0.5))
        cis = [c for c in curves if c.branch == "cislunar"]
        tra = [c for c in curves if c.branch == "translunar"]
        assert {c.formula_id for c in cis} == {"Eq.76"}
        assert {c.formula_id for c in tra} == {"Eq.78"}
        for curve in tra:
            finite = np.isfinite(curve.inclination_rad)
            assert finite.any()
            # 驻定倾角落在逆 Kozai 带：下界 63.4°（式 80），上界为该 e 切片
            # 的日外极限 acos(sqrt((1−e²)/5))（式 81）加数值余量。
            degrees = np.degrees(curve.inclination_rad[finite])
            upper_bound = math.degrees(math.acos(math.sqrt((1.0 - curve.eccentricity**2) / 5.0)))
            assert degrees.min() > 63.0
            assert degrees.max() < upper_bound + 0.1
        with pytest.raises(ValueError, match="branch"):
            s.secular_loci_curves(branches=("bogus",))  # type: ignore[arg-type]


class TestVzlk:
    def test_critical_inclination_eq64(self):
        assert pytest.approx(39.2, abs=0.05) == s.VZLK_CRITICAL_INCLINATION_DEG
        assert pytest.approx(140.8, abs=0.05) == 180.0 - s.VZLK_CRITICAL_INCLINATION_DEG

    def test_frequency_and_timescale_eq69_eq71(self):
        a_km = 2.0e5
        nu = s.vzlk_frequency_rad_s(a_km)
        expected_nu = 0.75 * a_km**1.5 / math.sqrt(C.earth_gm) * s.vzlk_tidal_sum(C)
        assert nu == pytest.approx(expected_nu, rel=1e-14)
        t_days = s.vzlk_timescale_days(a_km)
        expected_t = (16.0 / 15.0) * math.sqrt(C.earth_gm) / a_km**1.5 / s.vzlk_tidal_sum(C)
        assert t_days == pytest.approx(expected_t / 86400.0, rel=1e-14)
        # 式 71 与式 70（1/ν）差固定因子 16/20。
        assert t_days * 86400.0 == pytest.approx(0.8 / nu, rel=1e-12)

    def test_validity_screening(self):
        inner = s.vzlk_validity(4.0e4)
        outer = s.vzlk_validity(3.0 * C.moon_a_km)
        assert inner.j2_suppressed is True  # r_L 内 J2 抑制
        assert inner.double_averaging_warning is False  # α = 0.10 远离 a☾
        assert outer.j2_suppressed is False
        assert outer.double_averaging_warning is True  # α = 3.0 > 0.8，双平均失效
        assert inner.alpha == pytest.approx(4.0e4 / C.moon_a_km)
        assert inner.j2_rate_ratio > 1.0  # J2 特征频率占优（抑制的量化口径）

    def test_portrait_separatrix_geometry(self):
        """c1 < 3/5：分离线（c2 = 0）在 ω = 90° 达 y = sqrt(5c1/3)，
        对应最大偏心率 e_max = sqrt(1 − 5c1/3)。"""
        portrait = s.vzlk_phase_portrait(0.3)
        assert portrait.has_separatrix
        assert portrait.e_max == pytest.approx(math.sqrt(1.0 - 0.5), rel=1e-12)
        separatrix = [poly for level, poly in portrait.curves if level == 0.0]
        assert separatrix
        y_expected = math.sqrt(5.0 * 0.3 / 3.0)
        best = min(min(abs(p[0] - 90.0) for p in poly) for poly in separatrix)
        assert best < 2.0  # 分离线确实穿过 ω ≈ 90° 附近
        y_at_90 = min((abs(p[0] - 90.0), p[1]) for poly in separatrix for p in poly)[1]
        assert y_at_90 == pytest.approx(y_expected, abs=5e-3)

    def test_portrait_circulation_above_critical(self):
        portrait = s.vzlk_phase_portrait(0.8)
        assert not portrait.has_separatrix
        assert math.isnan(portrait.e_max)

    def test_portrait_c2_is_conserved_on_curves(self):
        """等值线点回代式 68 与等值一致（相图即轨迹）。"""
        c1 = 0.4
        portrait = s.vzlk_phase_portrait(c1, levels=(0.1,))
        for level, poly in portrait.curves:
            omega = np.radians(poly[:, 0])
            y = poly[:, 1]
            e2 = 1.0 - y**2
            sin2_i = 1.0 - c1 / y**2
            c2 = e2 * (0.4 - sin2_i * np.sin(omega) ** 2)
            np.testing.assert_allclose(c2, level, atol=2e-3)

    def test_c1_domain_validation(self):
        with pytest.raises(ValueError, match="c1"):
            s.vzlk_phase_portrait(0.0)
        with pytest.raises(ValueError, match="c1"):
            s.vzlk_phase_portrait(1.5)
