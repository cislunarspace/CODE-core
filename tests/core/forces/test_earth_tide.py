"""地球潮汐修正测试(Slice 10' / issue #108)。

迁移 GMAT HarmonicGravity 的固体潮 + 极潮 + tide-free/zero-tide 约定。
精度要求低,测试目标是覆盖迁移路径 + sanity check 量级。
"""

import numpy as np
import pytest

from e2m2e.core.forces.earth_tide import (
    permanent_tide_correction,
    pole_tide,
    solid_tide_step1,
    solid_tide_step2,
)


# 物理常量(量级参考用)
_MOON_MU = 4902.8001  # km³/s²
_EARTH_MU = 398600.4415  # km³/s²
_EARTH_R = 6378.1363  # km
_MOON_DIST = 384400.0  # km(近似)


class TestSolidTideStep1:
    """固体潮 Step 1(频率无关,Love 数 K/KPlus)。

    迁移 GMAT IncrementSolidTide:n=2..3, m=0..n,
    ΔC[n][m] += K[n][m]/(2n+1) * (μ_perturber/μ_earth) * (R_earth/r)^(n+1) * P_nm * cos(mλ)
    n=2 时额外 ΔC[4][m] += KPlus[m]/5 * ... (弹性 Love 数 3 阶位移)。
    """

    def test_delta_c20_magnitude_for_moon_on_x_axis(self):
        """月球在 x 轴(lat=0, lon=0)时 ΔC20 量级 ~1e-9。

        解析:ΔC20 = K20/5 * (μ_Moon/μ_Earth) * (R/r)³ * P20(0) * cos(0)
        P20(sinθ=0) = √5*(1.5*0-0.5) = -√5/2 < 0,故 ΔC20 < 0。
        """
        pos = np.array([_MOON_DIST, 0.0, 0.0])
        deltaC, deltaS = solid_tide_step1(
            pos, mu_perturber=_MOON_MU, mu_earth=_EARTH_MU, r_earth=_EARTH_R
        )

        # 量级 sanity:1e-10 到 1e-7
        assert 1e-10 < abs(deltaC[2, 0]) < 1e-7
        # P20(lat=0) < 0 → ΔC20 < 0
        assert deltaC[2, 0] < 0.0

    def test_delta_s20_zero_for_m_zero(self):
        """m=0 时 ΔS[2][0]=0(sin(0·λ)=0)。"""
        pos = np.array([_MOON_DIST, 0.0, 0.0])
        _, deltaS = solid_tide_step1(
            pos, mu_perturber=_MOON_MU, mu_earth=_EARTH_MU, r_earth=_EARTH_R
        )

        assert deltaS[2, 0] == pytest.approx(0.0, abs=1e-30)

    def test_delta_cs_shape_is_5x5(self):
        """返回 5×5 数组(GMAT LoveMax+1=5,覆盖 n=0..4)。"""
        pos = np.array([_MOON_DIST, 0.0, 0.0])
        deltaC, deltaS = solid_tide_step1(
            pos, mu_perturber=_MOON_MU, mu_earth=_EARTH_MU, r_earth=_EARTH_R
        )

        assert deltaC.shape == (5, 5)
        assert deltaS.shape == (5, 5)

    def test_delta_c22_nonzero_for_off_axis_moon(self):
        """月球偏离 x 轴(有经度)时 ΔC22 非零。"""
        pos = np.array([_MOON_DIST * 0.7, _MOON_DIST * 0.7, 0.0])  # lon=45°
        deltaC, deltaS = solid_tide_step1(
            pos, mu_perturber=_MOON_MU, mu_earth=_EARTH_MU, r_earth=_EARTH_R
        )

        # lon=45°, m=2: cos(2·45°)=cos(90°)=0 → ΔC22≈0; sin(90°)=1 → ΔS22≠0
        assert abs(deltaS[2, 2]) > 1e-12

    def test_delta_scales_with_perturber_mass_ratio(self):
        """ΔC 与扰动天体 GM 成正比(Sun 比 Moon 贡献大但距离远)。"""
        pos = np.array([_MOON_DIST, 0.0, 0.0])
        deltaC_moon, _ = solid_tide_step1(
            pos, mu_perturber=_MOON_MU, mu_earth=_EARTH_MU, r_earth=_EARTH_R
        )
        # GM 翻倍 → ΔC 翻倍(线性)
        deltaC_double, _ = solid_tide_step1(
            pos, mu_perturber=2.0 * _MOON_MU, mu_earth=_EARTH_MU, r_earth=_EARTH_R
        )

        np.testing.assert_allclose(deltaC_double[2, 0], 2.0 * deltaC_moon[2, 0], rtol=1e-12)


class TestSolidTideStep2:
    """固体潮 Step 2(频率相关,迁移 GMAT IncrementEarthTide 的 Delaunay 幅角段)。

    5 个 Delaunay 幅角 F[0..4] + GMST + Table6.3a/b/c,只影响 (2,0)/(2,1)/(2,2)。
    量级 ~1e-10(GMAT freq_dep * 1e-12 缩放)。
    """

    def test_delta_c20_nonzero_and_reasonable_at_j2000(self):
        """J2000 时刻 ΔC20(频率相关)非零,量级在 1e-12 到 1e-9。"""
        deltaC, _ = solid_tide_step2(et=0.0)

        assert deltaC[2, 0] != 0.0
        assert 1e-13 < abs(deltaC[2, 0]) < 1e-9

    def test_only_degree2_terms_nonzero(self):
        """Step 2 只写 (2,0)/(2,1)/(2,2),其余为零。"""
        deltaC, deltaS = solid_tide_step2(et=0.0)

        # degree 0,1,3,4 的 (n,0) 为零
        assert deltaC[0, 0] == 0.0
        assert deltaC[1, 0] == 0.0
        assert deltaC[3, 0] == 0.0
        assert deltaC[4, 0] == 0.0  # Step2 不写 (4,m);Step1 才写

    def test_delta_varies_with_time(self):
        """潮汐频率相关项随时间变化(Delaunay 幅角时变)。"""
        d1, _ = solid_tide_step2(et=0.0)
        d2, _ = solid_tide_step2(et=6 * 86400.0)  # 6 天后

        assert not np.allclose(d1[2, 0], d2[2, 0], atol=1e-15)

    def test_returns_5x5_arrays(self):
        deltaC, deltaS = solid_tide_step2(et=0.0)
        assert deltaC.shape == (5, 5)
        assert deltaS.shape == (5, 5)


class TestPoleTide:
    """极潮(固体极潮 IERS p.65 + Desai 海洋极潮 TN32 §6.3)。

    迁移 GMAT IncrementEarthTide 的极潮段:只影响 (2,1)。
    m1 = xp - xp_bar;m2 = -(yp - yp_bar);xp_bar/yp_bar 是 IERS p.84 长期漂移模型。
    量级 ~1e-9(GMAT 系数 1.333e-9 / 2.2344e-10)。
    xp/yp 单位 arcsec。
    """

    def test_only_21_nonzero(self):
        """极潮只影响 (2,1)。"""
        deltaC, deltaS = pole_tide(et=0.0, xp=0.1, yp=0.3)

        for n in range(5):
            for m in range(n + 1):
                if (n, m) != (2, 1):
                    assert deltaC[n, m] == 0.0
                    assert deltaS[n, m] == 0.0

    def test_delta_21_reasonable_magnitude(self):
        """ΔC21/ΔS21 量级在 1e-11 到 1e-8。"""
        deltaC, deltaS = pole_tide(et=0.0, xp=0.1, yp=0.3)

        assert 1e-12 < abs(deltaC[2, 1]) < 1e-8
        assert 1e-12 < abs(deltaS[2, 1]) < 1e-8

    def test_nonzero_even_when_xp_yp_zero(self):
        """xp=yp=0 时仍有长期漂移贡献(xp_bar≈0.054, yp_bar≈0.357 非零)。"""
        deltaC, deltaS = pole_tide(et=0.0, xp=0.0, yp=0.0)

        assert deltaS[2, 1] != 0.0

    def test_returns_5x5_arrays(self):
        deltaC, deltaS = pole_tide(et=0.0, xp=0.0, yp=0.0)
        assert deltaC.shape == (5, 5)
        assert deltaS.shape == (5, 5)


# 永久潮汐修正用物理常量(Sun/Moon 半长轴)
_SUN_MU = 1.32712440018e11  # km³/s²
_A_SUN = 1.495978707e8  # km
_A_MOON = _MOON_DIST  # km


class TestPermanentTideCorrection:
    """永久潮汐修正(IERS TN32 Step 3,时间平均,AC3)。

    zero-tide 系数约定下,GravityField 减去此值(系数已含永久潮汐)。
    用 solid_tide_step1 在 Sun/Moon 半长轴 + 赤道(零纬度,时间平均近似)计算。
    """

    def test_returns_5x5_nonzero_c20(self):
        dC, dS = permanent_tide_correction(
            mu_sun=_SUN_MU,
            mu_moon=_MOON_MU,
            mu_earth=_EARTH_MU,
            r_earth=_EARTH_R,
            a_sun=_A_SUN,
            a_moon=_A_MOON,
        )

        assert dC.shape == (5, 5)
        assert dC[2, 0] != 0.0

    def test_c20_magnitude_reasonable(self):
        """永久潮汐 ΔC20 量级 ~1e-9(Sun+Moon 在平均位置)。"""
        dC, _ = permanent_tide_correction(
            mu_sun=_SUN_MU,
            mu_moon=_MOON_MU,
            mu_earth=_EARTH_MU,
            r_earth=_EARTH_R,
            a_sun=_A_SUN,
            a_moon=_A_MOON,
        )

        assert 1e-10 < abs(dC[2, 0]) < 1e-7

    def test_moon_dominates_over_sun(self):
        """Moon 近,永久潮汐贡献大于 Sun。"""
        dC_sun_only, _ = permanent_tide_correction(
            _SUN_MU, 0.0, _EARTH_MU, _EARTH_R, _A_SUN, _A_MOON
        )
        dC_moon_only, _ = permanent_tide_correction(
            0.0, _MOON_MU, _EARTH_MU, _EARTH_R, _A_SUN, _A_MOON
        )

        assert abs(dC_moon_only[2, 0]) > abs(dC_sun_only[2, 0])
