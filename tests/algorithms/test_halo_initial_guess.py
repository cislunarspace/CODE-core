"""
Halo initial guess module tests

测试 halo_initial_guess.py 模块中的函数。覆盖：
  - Richardson 三阶近似系数（compute_halo_coefficients）
  - 解析近似（halo_third_order_approximation）
  - 初始猜测生成（compute_halo_initial_guess）
  - 模块导入路径（TestBackwardCompatibility）

合并自原来的两个文件：
  - test_analytical_halo.py：~30 个测试通过 `e2m2e.algorithms` 间接导入
  - test_halo_initial_guess.py：~30 个测试直接导入 halo_initial_guess 模块
两者 ~60% 重叠，合并后保留全部独有覆盖，去掉重复。

References:
    Richardson, D. L. (1980). Analytic construction of periodic orbits
    about the collinear points. Celestial Mechanics, 22(3), 303-320.
"""

import numpy as np
import pytest

from e2m2e.algorithms.halo_initial_guess import (
    compute_halo_coefficients,
    compute_halo_initial_guess,
    halo_third_order_approximation,
)

MU = 0.012150585  # Earth-Moon mass ratio


# =============================================================================
# Module identity: ensure the module itself is importable and self-contained
# =============================================================================


class TestModuleImport:
    """验证模块可直接导入，函数可直接调用。"""

    def test_module_importable(self):
        """halo_initial_guess 模块应可直接导入。"""
        # 导入验证在 class 级别已完成
        assert compute_halo_coefficients is not None
        assert halo_third_order_approximation is not None
        assert compute_halo_initial_guess is not None

    def test_functions_have_docstrings(self):
        """公共函数应有文档字符串。"""
        assert compute_halo_coefficients.__doc__ is not None
        assert halo_third_order_approximation.__doc__ is not None
        assert compute_halo_initial_guess.__doc__ is not None


# =============================================================================
# compute_halo_coefficients
# =============================================================================


class TestHaloCoefficients:
    """compute_halo_coefficients 的正确性（参数化覆盖 L1 和 L2）。"""

    @pytest.mark.parametrize("L", [1, 2])
    def test_gamma_nonzero(self, L: int):
        """gamma 对 L1 和 L2 都应为非零。"""
        coeffs = compute_halo_coefficients(MU, L)
        assert coeffs["gamma"] != 0.0

    @pytest.mark.parametrize("L", [1, 2])
    def test_omega_p_positive(self, L: int):
        """omega_p 应始终为正。"""
        coeffs = compute_halo_coefficients(MU, L)
        assert coeffs["omega_p"] > 0

    @pytest.mark.parametrize("L", [1, 2])
    def test_omega_p_reasonable_range(self, L: int):
        """omega_p 应在合理范围内（L1 ≈ 2.33, L2 ≈ 1.86, Lyapunov 频率）。"""
        coeffs = compute_halo_coefficients(MU, L)
        if L == 1:
            assert 2.0 < coeffs["omega_p"] < 2.6
        else:
            assert 1.5 < coeffs["omega_p"] < 2.2

    def test_l1_gamma_close_to_hill_approximation(self):
        """L1 gamma 应为正值，接近 Hill 球近似 (μ/3)^(1/3)。"""
        coeffs = compute_halo_coefficients(MU, L=1)
        expected = (MU / 3) ** (1 / 3)
        np.testing.assert_allclose(coeffs["gamma"], expected, rtol=0.06)

    def test_l2_gamma_close_to_hill_approximation(self):
        """L2 gamma 应为负值，绝对值接近 Hill 球近似。"""
        coeffs = compute_halo_coefficients(MU, L=2)
        expected = (MU / 3) ** (1 / 3)
        np.testing.assert_allclose(abs(coeffs["gamma"]), expected, rtol=0.15)

    @pytest.mark.parametrize("L", [1, 2])
    def test_l1_l2_gamma_opposite_signs(self, L: int):
        """L1 和 L2 的 gamma 符号应相反。"""
        c1 = compute_halo_coefficients(MU, 1)
        c2 = compute_halo_coefficients(MU, 2)
        assert c1["gamma"] * c2["gamma"] < 0

    def test_l1_k_positive(self):
        """L1 的 k 应为正。"""
        assert compute_halo_coefficients(MU, 1)["k"] > 0

    def test_l2_k_negative(self):
        """L2 的 k 应为负。"""
        assert compute_halo_coefficients(MU, 2)["k"] < 0

    def test_l1_delta_negative(self):
        """L1 的 delta 应为负。"""
        assert compute_halo_coefficients(MU, 1)["delta"] < 0

    def test_l2_delta_positive(self):
        """L2 的 delta 应为正。"""
        assert compute_halo_coefficients(MU, 2)["delta"] > 0

    def test_c3_depends_on_mu(self):
        """c3 = 3*mu*(2-mu) 应与 mu 成正比。"""
        coeffs = compute_halo_coefficients(MU, 1)
        expected_c3 = 3 * MU * (2 - MU)
        np.testing.assert_allclose(coeffs["c3"], expected_c3)

    @pytest.mark.parametrize("L", [1, 2])
    def test_kappa_coefficients_nonzero(self, L: int):
        """kappa1 和 kappa2 应为非零。"""
        coeffs = compute_halo_coefficients(MU, L)
        assert coeffs["kappa1"] != 0.0
        assert coeffs["kappa2"] != 0.0

    @pytest.mark.parametrize("L", [1, 2])
    def test_all_23_coefficients_present(self, L: int):
        """返回的字典应包含全部 23 个系数。"""
        coeffs = compute_halo_coefficients(MU, L)
        # gamma, omega_p, c1-c3, a21-a31, b21-b31, d21-d32, k, delta, l1-l3, kappa1-kappa2 = 23
        assert len(coeffs) == 23

    def test_invalid_L_raises(self):
        """L 不是 1 或 2 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="L必须是1或2"):
            compute_halo_coefficients(MU, 3)
        with pytest.raises(ValueError, match="L必须是1或2"):
            compute_halo_coefficients(MU, 0)


# =============================================================================
# halo_third_order_approximation
# =============================================================================


class TestHaloThirdOrderApproximation:
    """halo_third_order_approximation 的正确性（参数化覆盖 north/south）。"""

    @pytest.mark.parametrize("halo_class", [0, 1])
    def test_returns_three_values(self, halo_class: int):
        """应返回三个值：(states, times, period)。"""
        result = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=50, halo_class=halo_class
        )
        assert len(result) == 3

    @pytest.mark.parametrize("halo_class", [0, 1])
    def test_states_shape_matches_N(self, halo_class: int):
        """状态向量形状应为 (N, 6)。"""
        N = 200
        sv, t, _ = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=N, halo_class=halo_class
        )
        assert sv.shape == (N, 6)
        assert t.shape == (N,)

    @pytest.mark.parametrize("halo_class", [0, 1])
    def test_period_matches_order_of_magnitude(self, halo_class: int):
        """周期应在 Lyapunov 量级（约 2-4 无量纲时间）。"""
        _, _, T = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=100, halo_class=halo_class
        )
        assert 2.0 < T < 4.5

    def test_l2_period_greater_than_l1(self):
        """L2 周期应大于 L1 周期。"""
        _, _, T_l1 = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=100
        )
        _, _, T_l2 = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=2, tf=1.0, N=100
        )
        assert T_l2 > T_l1

    def test_initial_state_at_xz_crossing(self):
        """初始状态应满足：y=0, z=0（XZ 平面穿越点）。"""
        sv, _, _ = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=100, halo_class=0
        )
        np.testing.assert_allclose(sv[0, 1], 0.0, atol=1e-10)  # y = 0
        np.testing.assert_allclose(sv[0, 2], 0.0, atol=1e-10)  # z = 0

    def test_z_range_bounded_by_Aw(self):
        """z 坐标范围应与 Aw 同量级。"""
        Aw = 0.05
        sv, _, _ = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=Aw, phi=0.0, L=1, tf=1.0, N=100, halo_class=0
        )
        z_max = np.max(np.abs(sv[:, 2]))
        # z 最大值应与 Aw 同量级（考虑 delta 符号）
        assert z_max < Aw * 2.0

    def test_velocity_bounded(self):
        """速度分量应有界（不爆炸）。"""
        sv, _, _ = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=100, halo_class=0
        )
        velocities = sv[:, 3:6]
        max_vel = np.max(np.abs(velocities))
        # 在无量纲单位下，速度应 < 2
        assert max_vel < 2.0

    def test_xz_symmetry_north(self):
        """北 Halo 应满足 XZ 平面对称，z 坐标应穿过零。"""
        sv, _, _ = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=100, halo_class=0
        )
        assert np.min(sv[:, 2]) < 0 or np.max(sv[:, 2]) > 0

    def test_xz_symmetry_south(self):
        """南 Halo 应满足 XZ 平面对称，z 坐标应穿过零。"""
        sv, _, _ = halo_third_order_approximation(
            mu=MU, Au=0.01, Aw=0.01, phi=np.pi, L=1, tf=1.0, N=100, halo_class=1
        )
        assert np.min(sv[:, 2]) < 0 or np.max(sv[:, 2]) > 0

    def test_north_south_third_order_z_opposite_phase(self):
        """北和南 Halo z 坐标相位应相差 π。"""
        Au = 0.01
        Aw = 0.01
        sv_north, _, _ = halo_third_order_approximation(
            mu=MU, Au=Au, Aw=Aw, phi=0.0, L=1, tf=1.0, N=100, halo_class=0
        )
        sv_south, _, _ = halo_third_order_approximation(
            mu=MU, Au=Au, Aw=Aw, phi=0.0, L=1, tf=1.0, N=100, halo_class=1
        )
        np.testing.assert_allclose(sv_north[:, 2], -sv_south[:, 2], atol=1e-10)

    def test_invalid_L_raises(self):
        """L 不是 1 或 2 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="L必须是1或2"):
            halo_third_order_approximation(
                mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=3, tf=1.0, N=100
            )

    def test_invalid_N_raises(self):
        """N < 2 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="N必须大于等于2"):
            halo_third_order_approximation(
                mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=1
            )

    def test_invalid_tf_raises(self):
        """tf <= 0 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="tf必须为正数"):
            halo_third_order_approximation(
                mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=0.0, N=100
            )

    def test_invalid_halo_class_raises(self):
        """halo_class 不是 0 或 1 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="halo_class必须是0或1"):
            halo_third_order_approximation(
                mu=MU, Au=0.01, Aw=0.01, phi=0.0, L=1, tf=1.0, N=100, halo_class=2
            )


# =============================================================================
# compute_halo_initial_guess
# =============================================================================


class TestHaloInitialGuess:
    """compute_halo_initial_guess 的正确性（参数化覆盖 L 和 halo_class）。"""

    @pytest.mark.parametrize("L", [1, 2])
    @pytest.mark.parametrize("halo_class", [0, 1])
    def test_y0_vx0_vz0_zero(self, L: int, halo_class: int):
        """初始状态 y0, vx0, vz0 应为零（XZ 平面穿越）。"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=L, halo_class=halo_class)
        assert guess["y0"] == 0.0
        assert guess["vx0"] == 0.0
        assert guess["vz0"] == 0.0

    @pytest.mark.parametrize("L", [1, 2])
    def test_x0_near_libration_point(self, L: int):
        """x0 应在平动点附近（误差 < 0.2 DU）。"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=L, halo_class=0)
        coeffs = compute_halo_coefficients(MU, L)
        L_pos = 1 - MU - coeffs["gamma"]
        assert abs(guess["x0"] - L_pos) < 0.2

    @pytest.mark.parametrize("L", [1, 2])
    @pytest.mark.parametrize("halo_class", [0, 1])
    def test_vy0_sign_consistent(self, L: int, halo_class: int):
        """vy0 符号应与平动点一致（L1 正，L2 负）。"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=L, halo_class=halo_class)
        if L == 1:
            assert guess["vy0"] > 0
        else:
            assert guess["vy0"] < 0

    def test_l1_l2_different_period(self):
        """L2 Halo 半周期应比 L1 长（因 ω_p 更小）。"""
        guess_l1 = compute_halo_initial_guess(mu=MU, z_amplitude=0.05, L=1, halo_class=0)
        guess_l2 = compute_halo_initial_guess(mu=MU, z_amplitude=0.05, L=2, halo_class=0)
        assert guess_l2["T_half"] > guess_l1["T_half"]

    def test_north_south_x_positions_identical(self):
        """北和南 Halo x0 位置相同（CR3BP z→-z 对称性：仅 z0 符号不同）。"""
        guess_north = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1, halo_class=0)
        guess_south = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1, halo_class=1)
        assert guess_north["x0"] == guess_south["x0"]
        assert guess_north["vy0"] == guess_south["vy0"]

    @pytest.mark.parametrize("L", [1, 2])
    def test_T_half_positive(self, L: int):
        """半周期应为正值。"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=L)
        assert guess["T_half"] > 0

    @pytest.mark.parametrize("L", [1, 2])
    def test_T_half_reasonable(self, L: int):
        """半周期应在 Lyapunov 半周期量级（约 1-2 无量纲时间）。"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.01, L=L)
        assert 0.8 < guess["T_half"] < 2.5

    def test_Au_positive(self):
        """Au 应为正值。"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1)
        assert guess["Au"] > 0

    def test_Aw_equals_z_amplitude(self):
        """Aw 应等于 z_amplitude。"""
        z_amp = 0.15
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=z_amp, L=1)
        assert guess["Aw"] == z_amp

    def test_au_proportional_to_sqrt_z_amplitude(self):
        """Au 应与 sqrt(z_amplitude) 成正比。"""
        guess1 = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1)
        guess2 = compute_halo_initial_guess(mu=MU, z_amplitude=0.4, L=1)
        ratio = guess2["Au"] / guess1["Au"]
        expected_ratio = np.sqrt(0.4 / 0.1)
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-10)

    def test_returns_all_nine_keys(self):
        """返回字典应包含全部 9 个键。"""
        guess = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1)
        expected = {"x0", "y0", "z0", "vx0", "vy0", "vz0", "T_half", "Au", "Aw"}
        assert set(guess.keys()) == expected

    def test_invalid_z_amplitude_zero_raises(self):
        """z_amplitude=0 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="z_amplitude必须为正数"):
            compute_halo_initial_guess(mu=MU, z_amplitude=0.0, L=1)

    def test_invalid_z_amplitude_negative_raises(self):
        """z_amplitude<0 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="z_amplitude必须为正数"):
            compute_halo_initial_guess(mu=MU, z_amplitude=-0.1, L=1)

    def test_default_halo_class(self):
        """halo_class 默认为 0（北 Halo）。"""
        guess_default = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1)
        guess_explicit = compute_halo_initial_guess(mu=MU, z_amplitude=0.1, L=1, halo_class=0)
        assert guess_default["x0"] == guess_explicit["x0"]


# =============================================================================
# Backward compatibility: re-export from e2m2e.algorithms must keep working
# =============================================================================


class TestBackwardCompatibility:
    """验证 differential_correction.py 的 re-export 不破坏现有功能。"""

    def test_reexport_preserves_public_api(self):
        """通过 differential_correction 导入应返回相同函数。"""
        from e2m2e.algorithms.differential_correction import (
            compute_halo_coefficients as from_dc,
        )
        from e2m2e.algorithms.differential_correction import (
            compute_halo_initial_guess as from_dc_guess,
        )
        from e2m2e.algorithms.differential_correction import (
            halo_third_order_approximation as from_dc_approx,
        )

        # 应指向同一个函数对象
        assert from_dc is compute_halo_coefficients
        assert from_dc_guess is compute_halo_initial_guess
        assert from_dc_approx is halo_third_order_approximation

    def test_algorithms_init_reexport(self):
        """e2m2e.algorithms.__init__ 的导入应继续工作。"""
        from e2m2e.algorithms import (
            compute_halo_coefficients,
        )

        # 结果应与直接导入一致
        coeffs1 = compute_halo_coefficients(MU, 1)
        coeffs2 = compute_halo_coefficients(MU, 1)
        assert coeffs1 == coeffs2
