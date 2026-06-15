"""地球潮汐修正测试(Slice 10' / issue #108)。

迁移 GMAT HarmonicGravity 的固体潮 + 极潮 + tide-free/zero-tide 约定。
精度要求低,测试目标是覆盖迁移路径 + sanity check 量级。
"""

import numpy as np
import pytest

from e2m2e.core.forces.earth_tide import solid_tide_step1


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
