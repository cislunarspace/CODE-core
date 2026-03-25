"""
Halo轨道解析近似模块测试

测试Richardson三阶近似等解析方法生成Halo轨道初始猜测的正确性。

参考论文：
  Richardson, D. L. (1980). Analytic construction of periodic orbits
  about the collinear points. Celestial Mechanics.

地月系统参数：
  μ = 1.2150585 × 10⁻²
  DU = 3.84405 × 10⁵ km, TU = 4.34811305 天
"""

import numpy as np
import pytest

from e2m2e.algorithms import (
    compute_halo_coefficients,
    halo_third_order_approximation,
    compute_halo_initial_guess,
)

MU = 0.012150585


class TestHaloCoefficients:
    """测试 compute_halo_coefficients 函数"""

    def test_l1_coefficients_gamma(self):
        """L1 Halo 系数 gamma 应为正值"""
        coeffs = compute_halo_coefficients(MU, L=1)
        assert coeffs["gamma"] > 0
        np.testing.assert_allclose(coeffs["gamma"], 0.012149, atol=1e-5)

    def test_l2_coefficients_gamma(self):
        """L2 Halo 系数 gamma 应为负值"""
        coeffs = compute_halo_coefficients(MU, L=2)
        assert coeffs["gamma"] < 0
        np.testing.assert_allclose(coeffs["gamma"], -0.012149, atol=1e-5)

    def test_l1_k_delta(self):
        """L1 Halo k=1, delta=-1 (Class I/North)"""
        coeffs = compute_halo_coefficients(MU, L=1)
        assert coeffs["k"] == 1.0
        assert coeffs["delta"] == -1.0

    def test_l2_k_delta(self):
        """L2 Halo k=-1, delta=1 (Class I/North)"""
        coeffs = compute_halo_coefficients(MU, L=2)
        assert coeffs["k"] == -1.0
        assert coeffs["delta"] == 1.0

    def test_l1_l2_different_signs(self):
        """L1 和 L2 的 gamma 符号应相反"""
        coeffs_l1 = compute_halo_coefficients(MU, L=1)
        coeffs_l2 = compute_halo_coefficients(MU, L=2)
        assert coeffs_l1["gamma"] == -coeffs_l2["gamma"]

    def test_invalid_l_throws(self):
        """L 不是 1 或 2 时应抛出 ValueError"""
        with pytest.raises(ValueError, match="L必须是1或2"):
            compute_halo_coefficients(MU, L=3)

    def test_returns_all_required_keys(self):
        """返回的字典应包含所有必需的系数"""
        coeffs = compute_halo_coefficients(MU, L=1)
        required_keys = [
            "gamma",
            "c1",
            "c2",
            "c3",
            "a21",
            "a22",
            "a23",
            "a24",
            "a31",
            "b21",
            "b22",
            "b31",
            "d21",
            "d31",
            "d32",
            "k",
            "delta",
            "l1",
            "l2",
            "l3",
            "kappa1",
            "kappa2",
        ]
        for key in required_keys:
            assert key in coeffs, f"Missing key: {key}"


class TestHaloThirdOrderApproximation:
    """测试 halo_third_order_approximation 函数"""

    def test_output_shape(self):
        """输出状态向量形状应为 (N, 6)"""
        Au = 0.01
        Aw = 0.01
        sv, t, T = halo_third_order_approximation(
            mu=MU, Au=Au, Aw=Aw, phi=0.0, L=1, tf=1.0, N=100, halo_class=0
        )
        assert sv.shape == (100, 6)
        assert t.shape == (100,)

    def test_period_positive(self):
        """计算的周期应为正值"""
        _, _, T = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=100, halo_class=0
        )
        assert T > 0

    def test_l1_halo_period_near_half_dimensionless(self):
        """L1 Halo 周期应接近 0.5 无量纲周期（约 2.77 天）"""
        Au = 0.01
        Aw = 0.01
        _, _, T = halo_third_order_approximation(
            mu=MU, Au=Au, Aw=Aw, phi=0.0, L=1, tf=1.0, N=100, halo_class=0
        )
        assert 0.4 < T < 0.6

    def test_l2_halo_period_similar_to_l1(self):
        """L2 Halo 周期应与 L1 相近"""
        _, _, T_l1 = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=100, halo_class=0
        )
        _, _, T_l2 = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=2, tf=1.0, N=100, halo_class=0
        )
        np.testing.assert_allclose(T_l1, T_l2, rtol=0.05)

    def test_north_halo_z_changes_sign(self):
        """北 Halo (Class I) z 坐标应改变符号"""
        Au = 0.01
        Aw = 0.01
        sv, _, _ = halo_third_order_approximation(
            mu=MU, Au=Au, Aw=Aw, phi=0.0, L=1, tf=1.0, N=100, halo_class=0
        )
        z = sv[:, 2]
        assert np.min(z) < 0 or np.max(z) > 0

    def test_south_halo_z_changes_sign(self):
        """南 Halo (Class II) z 坐标应改变符号"""
        Au = 0.01
        Aw = 0.01
        sv, _, _ = halo_third_order_approximation(
            mu=MU, Au=Au, Aw=Aw, phi=np.pi, L=1, tf=1.0, N=100, halo_class=1
        )
        z = sv[:, 2]
        assert np.min(z) < 0 or np.max(z) > 0

    def test_xz_symmetry_at_start(self):
        """初始状态应满足 XZ 平面对称条件: y=0, z=0"""
        sv, _, _ = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=100, halo_class=0
        )
        np.testing.assert_allclose(sv[0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(sv[0, 2], 0.0, atol=1e-10)

    def test_invalid_l_throws(self):
        """L 不是 1 或 2 时应抛出 ValueError"""
        with pytest.raises(ValueError, match="L必须是1或2"):
            halo_third_order_approximation(
                mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=3, tf=1.0, N=100, halo_class=0
            )

    def test_invalid_n_throws(self):
        """N < 2 时应抛出 ValueError"""
        with pytest.raises(ValueError, match="N必须大于等于2"):
            halo_third_order_approximation(
                mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=1, halo_class=0
            )

    def test_invalid_tf_throws(self):
        """tf <= 0 时应抛出 ValueError"""
        with pytest.raises(ValueError, match="tf必须为正数"):
            halo_third_order_approximation(
                mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=0.0, N=100, halo_class=0
            )

    def test_invalid_halo_class_throws(self):
        """halo_class 不是 0 或 1 时应抛出 ValueError"""
        with pytest.raises(ValueError, match="halo_class必须是0或1"):
            halo_third_order_approximation(
                mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=100, halo_class=2
            )


class TestHaloInitialGuess:
    """测试 compute_halo_initial_guess 函数"""

    def test_l1_north_initial_state(self):
        """L1 北 Halo 初始状态应满足对称条件"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1, halo_class=0)
        assert guess["y0"] == 0.0
        assert guess["z0"] == 0.0
        assert guess["vx0"] == 0.0
        assert guess["vz0"] == 0.0

    def test_l2_north_initial_state(self):
        """L2 北 Halo 初始状态应满足对称条件"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=2, halo_class=0)
        assert guess["y0"] == 0.0
        assert guess["z0"] == 0.0
        assert guess["vx0"] == 0.0
        assert guess["vz0"] == 0.0

    def test_l1_x_position(self):
        """L1 Halo x0 应位于 L1 点左侧（< 1-mu-gamma）"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1, halo_class=0)
        L1_position = 1 - MU - 0.012149
        assert guess["x0"] < L1_position

    def test_l2_x_position(self):
        """L2 Halo x0 应位于 L2 点右侧（> 1-mu+gamma）"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=2, halo_class=0)
        L2_position = 1 - MU + 0.012149
        assert guess["x0"] > L2_position

    def test_vy0_positive_for_l1_north(self):
        """L1 北 Halo vy0 应为正"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1, halo_class=0)
        assert guess["vy0"] > 0

    def test_vy0_negative_for_l2_north(self):
        """L2 北 Halo vy0 应为负"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=2, halo_class=0)
        assert guess["vy0"] < 0

    def test_half_period_positive(self):
        """半周期应为正值"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1, halo_class=0)
        assert guess["T_half"] > 0

    def test_half_period_near_pi(self):
        """半周期应接近 π（无量纲时间）"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.01, L=1, halo_class=0)
        np.testing.assert_allclose(guess["T_half"], np.pi, rtol=0.1)

    def test_amplitude_relationship(self):
        """Au 和 Aw (z_amplitude) 应满足振幅关系"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1, halo_class=0)
        assert "Au" in guess
        assert "Aw" in guess
        assert guess["Aw"] == 0.1
        assert guess["Au"] > 0

    def test_invalid_z_amplitude_throws(self):
        """z_amplitude <= 0 时应抛出 ValueError"""
        with pytest.raises(ValueError, match="z_amplitude必须为正数"):
            compute_halo_initial_guess(mu=MU, z_amplitude=0.0, L=1, halo_class=0)

        with pytest.raises(ValueError, match="z_amplitude必须为正数"):
            compute_halo_initial_guess(mu=MU, z_amplitude=-0.1, L=1, halo_class=0)

    def test_returns_all_required_keys(self):
        """返回的字典应包含所有必需的键"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1, halo_class=0)
        required_keys = ["x0", "y0", "z0", "vx0", "vy0", "vz0", "T_half", "Au", "Aw"]
        for key in required_keys:
            assert key in guess, f"Missing key: {key}"


class TestL1L2HaloPeriod:
    """测试 L1/L2 Halo 周期计算"""

    def test_l1_l2_periods_similar(self):
        """L1 和 L2 Halo 周期应相近"""
        guess_l1 = compute_halo_initial_guess(mu=MU, z_amplitude=0.05, L=1, halo_class=0)
        guess_l2 = compute_halo_initial_guess(mu=MU, z_amplitude=0.05, L=2, halo_class=0)
        np.testing.assert_allclose(guess_l1["T_half"], guess_l2["T_half"], rtol=0.05)

    def test_larger_amplitude_longer_period(self):
        """较大振幅应有较长的周期"""
        guess_small = compute_halo_initial_guess(mu=MU, z_amplitude=0.05, L=1, halo_class=0)
        guess_large = compute_halo_initial_guess(mu=MU, z_amplitude=0.15, L=1, halo_class=0)
        assert guess_large["T_half"] > guess_small["T_half"]


class TestNorthSouthHalo:
    """测试北/南 Halo (Class I/II) 的差异"""

    def test_north_south_initial_y_symmetric(self):
        """北和南 Halo 初始 y 状态相同"""
        guess_north = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1, halo_class=0)
        guess_south = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1, halo_class=1)
        assert guess_north["y0"] == guess_south["y0"]
        assert guess_north["vy0"] == -guess_south["vy0"]

    def test_north_south_x_positions_differ(self):
        """北和南 Halo x0 位置应略有不同"""
        guess_north = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1, halo_class=0)
        guess_south = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1, halo_class=1)
        assert guess_north["x0"] != guess_south["x0"]

    def test_north_south_third_order_z_opposite_phase(self):
        """北和南 Halo z 坐标相位应相差 π"""
        Au = 0.01
        Aw = 0.01
        sv_north, _, _ = halo_third_order_approximation(
            mu=MU, Au=Au, Aw=Aw, phi=0.0, L=1, tf=1.0, N=100, halo_class=0
        )
        sv_south, _, _ = halo_third_order_approximation(
            mu=MU, Au=Au, Aw=Aw, phi=0.0, L=1, tf=1.0, N=100, halo_class=1
        )
        np.testing.assert_allclose(sv_north[:, 2], -sv_south[:, 2], atol=1e-10)


class TestAmplitudeRelationships:
    """测试振幅关系"""

    def test_au_proportional_to_sqrt_z_amplitude(self):
        """Au 应与 sqrt(z_amplitude) 成正比"""
        coeffs_l1 = compute_halo_coefficients(MU, L=1)
        kappa1 = coeffs_l1["kappa1"]
        l1 = coeffs_l1["l1"]
        z1 = 0.05
        z2 = 0.20
        au1 = np.sqrt(-kappa1 * z1**2 / l1)
        au2 = np.sqrt(-kappa1 * z2**2 / l1)
        expected_ratio = np.sqrt(z2 / z1)
        np.testing.assert_allclose(au2 / au1, expected_ratio, rtol=1e-10)
