"""固体潮修正测试。

覆盖固体潮 Step1(天体无关)/Step2(地球专用)、极潮、tide-free/zero-tide 约定,
以及月球固体潮(Love 数取自 grgm900c.tide)。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces.earth_tide import (
    _K_EARTH,
    _K_PLUS_EARTH,
    load_love_number_file,
    permanent_tide_correction,
    pole_tide,
    solid_tide_step1,
    solid_tide_step2,
)

pytestmark = pytest.mark.force


# 物理常量(量级参考用)
_MOON_MU = 4902.8001  # km³/s²
_EARTH_MU = 398600.4415  # km³/s²
_EARTH_R = 6378.1363  # km
_MOON_DIST = 384400.0  # km(近似)


def _earth_step1_single(pos, mu_perturber):
    """用地球 Love 数对单个扰动体跑 solid_tide_step1(测试便捷封装)。"""
    return solid_tide_step1(
        [(pos, mu_perturber)],
        k_love=_K_EARTH,
        k_plus=_K_PLUS_EARTH,
        mu_central=_EARTH_MU,
        r_central=_EARTH_R,
    )


class TestSolidTideStep1:
    """固体潮 Step 1(天体无关,Love 数 K/KPlus 由调用方传入)。

    公式(IERS TN32 eqn 1):n=2..3, m=0..n,
    ΔC[n][m] += K[n][m]/(2n+1) * (μ_perturber/μ_central) * (R/r)^(n+1) * P_nm * cos(mλ)
    n=2 时额外 ΔC[4][m] += KPlus[m]/5 * ... (弹性 Love 数 3 阶位移)。
    """

    def test_delta_c20_magnitude_for_moon_on_x_axis(self):
        """月球在 x 轴(lat=0, lon=0)时 ΔC20 量级 ~1e-9。

        解析:ΔC20 = K20/5 * (μ_Moon/μ_Earth) * (R/r)³ * P20(0) * cos(0)
        P20(sinθ=0) = √5*(1.5*0-0.5) = -√5/2 < 0,故 ΔC20 < 0。
        """
        pos = np.array([_MOON_DIST, 0.0, 0.0])
        deltaC, deltaS = _earth_step1_single(pos, _MOON_MU)

        # 量级 sanity:1e-10 到 1e-7
        assert 1e-10 < abs(deltaC[2, 0]) < 1e-7
        # P20(lat=0) < 0 → ΔC20 < 0
        assert deltaC[2, 0] < 0.0

    def test_delta_s20_zero_for_m_zero(self):
        """m=0 时 ΔS[2][0]=0(sin(0·λ)=0)。"""
        pos = np.array([_MOON_DIST, 0.0, 0.0])
        _, deltaS = _earth_step1_single(pos, _MOON_MU)

        assert deltaS[2, 0] == pytest.approx(0.0, abs=1e-30)

    def test_delta_cs_shape_is_5x5(self):
        """返回 5×5 数组(GMAT LoveMax+1=5,覆盖 n=0..4)。"""
        pos = np.array([_MOON_DIST, 0.0, 0.0])
        deltaC, deltaS = _earth_step1_single(pos, _MOON_MU)

        assert deltaC.shape == (5, 5)
        assert deltaS.shape == (5, 5)

    def test_delta_c22_nonzero_for_off_axis_moon(self):
        """月球偏离 x 轴(有经度)时 ΔC22 非零。"""
        pos = np.array([_MOON_DIST * 0.7, _MOON_DIST * 0.7, 0.0])  # lon=45°
        deltaC, deltaS = _earth_step1_single(pos, _MOON_MU)

        # lon=45°, m=2: cos(2·45°)=cos(90°)=0 → ΔC22≈0; sin(90°)=1 → ΔS22≠0
        assert abs(deltaS[2, 2]) > 1e-12

    def test_delta_scales_with_perturber_mass_ratio(self):
        """ΔC 与扰动天体 GM 成正比(Sun 比 Moon 贡献大但距离远)。"""
        pos = np.array([_MOON_DIST, 0.0, 0.0])
        deltaC_moon, _ = _earth_step1_single(pos, _MOON_MU)
        # GM 翻倍 → ΔC 翻倍(线性)
        deltaC_double, _ = _earth_step1_single(pos, 2.0 * _MOON_MU)

        np.testing.assert_allclose(deltaC_double[2, 0], 2.0 * deltaC_moon[2, 0], rtol=1e-12)

    def test_single_tuple_perturber_accepted(self):
        """单个 (position, gm) 元组也应被接受(兼容旧调用方)。"""
        pos = np.array([_MOON_DIST, 0.0, 0.0])
        dC_list, _ = solid_tide_step1(
            [(pos, _MOON_MU)],
            k_love=_K_EARTH,
            k_plus=_K_PLUS_EARTH,
            mu_central=_EARTH_MU,
            r_central=_EARTH_R,
        )
        dC_tuple, _ = solid_tide_step1(
            (pos, _MOON_MU),
            k_love=_K_EARTH,
            k_plus=_K_PLUS_EARTH,
            mu_central=_EARTH_MU,
            r_central=_EARTH_R,
        )
        np.testing.assert_allclose(dC_list, dC_tuple, atol=0.0)

    def test_accumulates_multiple_perturbers(self):
        """多个扰动体(Sun+Moon)累加 = 各自单独贡献之和。"""
        sun_pos = np.array([1.495978707e8, 0.0, 0.0])
        sun_mu = 1.32712440018e11
        moon_pos = np.array([_MOON_DIST, 0.0, 0.0])
        dC_both, dS_both = solid_tide_step1(
            [(sun_pos, sun_mu), (moon_pos, _MOON_MU)],
            k_love=_K_EARTH,
            k_plus=_K_PLUS_EARTH,
            mu_central=_EARTH_MU,
            r_central=_EARTH_R,
        )
        dC_sun, dS_sun = _earth_step1_single(sun_pos, sun_mu)
        dC_moon, dS_moon = _earth_step1_single(moon_pos, _MOON_MU)
        np.testing.assert_allclose(dC_both, dC_sun + dC_moon, atol=0.0)
        np.testing.assert_allclose(dS_both, dS_sun + dS_moon, atol=0.0)

    def test_k_plus_none_skips_degree4(self):
        """k_plus=None 时不写 n=4 项(月球等无弹性 3 阶位移)。"""
        pos = np.array([_MOON_DIST, 0.0, 0.0])
        dC, dS = solid_tide_step1(
            [(pos, _MOON_MU)],
            k_love=_K_EARTH,
            k_plus=None,
            mu_central=_EARTH_MU,
            r_central=_EARTH_R,
        )
        # n=4 全零
        assert np.all(dC[4, :] == 0.0)
        assert np.all(dS[4, :] == 0.0)
        # 但 n=2,3 仍正常计算
        assert dC[2, 0] != 0.0


# ----------------------------------------------------------------------------
# 月球固体潮:Love 数从 grgm900c.tide 读取(k₂=0.024116)
# ----------------------------------------------------------------------------

_MOON_R = 1738.0  # km(月球参考半径)
_K_MOON_2 = 0.024116  # grgm900c.tide 中 k₂₀=k₂₁=k₂₂


def _moon_k_love():
    """构造月球 Love 数表(仅 k₂₀=k₂₁=k₂₂=0.024116)。"""
    k = np.zeros((5, 5), dtype=float)
    k[2, 0] = k[2, 1] = k[2, 2] = _K_MOON_2
    return k


class TestMoonLoveNumberFile:
    """``load_love_number_file`` 解析 grgm900c.tide 格式。"""

    def test_load_grgm900c_tide(self, tmp_path):
        """读取包内 grgm900c.tide 得到 k₂=0.024116 三项。"""
        content = "% GMAT solid lunar tide model\nk 2 0 0.024116\nk 2 1 0.024116\nk 2 2 0.024116\n"
        path = tmp_path / "moon.tide"
        path.write_text(content)
        k = load_love_number_file(path)
        assert k.shape == (5, 5)
        assert k[2, 0] == pytest.approx(0.024116, rel=1e-12)
        assert k[2, 1] == pytest.approx(0.024116, rel=1e-12)
        assert k[2, 2] == pytest.approx(0.024116, rel=1e-12)
        # 其余为零
        assert k[0, 0] == 0.0
        assert k[3, 0] == 0.0

    def test_ignores_comments_and_blank_lines(self, tmp_path):
        """注释行(% 与 #)与空行被跳过。"""
        path = tmp_path / "c.tide"
        path.write_text("# header\n\n% another\nk 2 0 0.05\n")
        k = load_love_number_file(path)
        assert k[2, 0] == pytest.approx(0.05, rel=1e-12)


class TestMoonSolidTideStep1:
    """月球固体潮 Step1:扰动体=地球(相对月球),Love 数 k₂=0.024116。

    量级 ~1e-10(k₂ 是地球 0.30 的 ~1/12,且地球 GM 大但距离远)。
    """

    def test_moon_delta_c20_magnitude(self):
        """地球在月球 x 轴(赤道,经度 0)时 ΔC20 量级 ~1e-8。

        解析:ΔC20 = k20/5 * (μ_Earth/μ_Moon) * (R_Moon/r)³ * P20(0)
        μ_Earth/μ_Moon ≈ 81.3, R_Moon/r ≈ 1738/384400 ≈ 0.00452,
        k20/5 ≈ 0.0048, P20(0) = -√5/2 ≈ -1.118。
        全式 ≈ 0.0048 * 81.3 * 9.2e-8 * (-1.118) ≈ -4e-8。
        注:虽然 k₂ 月球(0.024)是地球(0.30)的 ~1/12,但地球作为月球扰动体
        时 massratio μ_Earth/μ_Moon ≈ 81 远大于地球潮汐的 μ_Moon/μ_Earth ≈ 0.012,
        故月球潮汐 ΔC 绝对量级仍可观(~1e-8)。
        """
        # 地球相对月球的位置(月心系,沿 +x)
        earth_pos = np.array([_MOON_DIST, 0.0, 0.0])
        dC, dS = solid_tide_step1(
            [(earth_pos, _EARTH_MU)],
            k_love=_moon_k_love(),
            k_plus=None,
            mu_central=_MOON_MU,
            r_central=_MOON_R,
        )
        # ΔC20 应非零且 < 0(P20(0)<0)
        assert dC[2, 0] != 0.0
        assert dC[2, 0] < 0.0
        # 量级:1e-9 到 1e-7
        assert 1e-9 < abs(dC[2, 0]) < 1e-7

    def test_moon_delta_smaller_than_earth_by_love_ratio(self):
        """控制变量:固定扰动体 GM + 中心天体 GM/R,仅换 Love 数表,
        月球 ΔC20 / 地球 ΔC20 = k₂_Moon / k₂_Earth ≈ 0.024/0.30 ≈ 0.08。"""
        pos = np.array([_MOON_DIST, 0.0, 0.0])
        # 同样扰动体/中心天体,仅 Love 数不同
        dC_earth, _ = solid_tide_step1(
            [(pos, _MOON_MU)],
            k_love=_K_EARTH,
            k_plus=_K_PLUS_EARTH,
            mu_central=_EARTH_MU,
            r_central=_EARTH_R,
        )
        dC_moon_love, _ = solid_tide_step1(
            [(pos, _MOON_MU)],
            k_love=_moon_k_love(),
            k_plus=None,
            mu_central=_EARTH_MU,
            r_central=_EARTH_R,
        )
        ratio = abs(dC_moon_love[2, 0]) / abs(dC_earth[2, 0])
        # k₂比例 ≈ 0.024116/0.30190 ≈ 0.0799
        expected_ratio = 0.024116 / 0.30190
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-6)

    def test_moon_no_degree4_contribution(self):
        """月球无弹性 3 阶位移(k_plus=None)→ n=4 全零。"""
        earth_pos = np.array([_MOON_DIST, 0.0, 0.0])
        dC, dS = solid_tide_step1(
            [(earth_pos, _EARTH_MU)],
            k_love=_moon_k_love(),
            k_plus=None,
            mu_central=_MOON_MU,
            r_central=_MOON_R,
        )
        assert np.all(dC[4, :] == 0.0)
        assert np.all(dS[4, :] == 0.0)

    def test_moon_hand_computed_c20(self):
        """手算验证:ΔC20 = k20/5 * massratio * rho³ * P20(0) * cos(0)。"""
        earth_pos = np.array([_MOON_DIST, 0.0, 0.0])
        r = _MOON_DIST
        massratio = _EARTH_MU / _MOON_MU
        rho = _MOON_R / r
        sqrt5 = np.sqrt(5.0)
        P20_0 = sqrt5 * (1.5 * 0.0 - 0.5)  # sin(lat)=0
        expected_C20 = (_K_MOON_2 / 5.0) * massratio * (rho**3) * P20_0 * np.cos(0.0)

        dC, _ = solid_tide_step1(
            [(earth_pos, _EARTH_MU)],
            k_love=_moon_k_love(),
            k_plus=None,
            mu_central=_MOON_MU,
            r_central=_MOON_R,
        )
        np.testing.assert_allclose(dC[2, 0], expected_C20, rtol=1e-12)


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
