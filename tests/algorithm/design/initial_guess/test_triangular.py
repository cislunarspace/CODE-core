"""L4/L5 三角平动点初猜测试（纯 CR3BP，不依赖 SPICE 内核）。"""

import numpy as np
import pytest

from e2m2e.algorithm.family.triangular_initial_guess import (
    _triangular_modes,
    compute_triangular_initial_guess,
)
from e2m2e.data.templates.enums import LibrationPoint

pytestmark = pytest.mark.orchestration


class TestTriangularModes:
    def test_frequencies_satisfy_characteristic_equation(self, earth_moon_system):
        """纯 CR3BP 下 ω_s/ω_l 满足 ω⁴ − ω² + 27μ(1−μ)/4 = 0。"""
        mu = earth_moon_system.mu
        for point in (4, 5):
            omega_s, _, omega_l, _, _omega_v, _, x_L = _triangular_modes(earth_moon_system, point)
            for omega in (omega_s, omega_l):
                residual = omega**4 - omega**2 + 27.0 * mu * (1.0 - mu) / 4.0
                assert abs(residual) < 1e-6, f"L{point} ω={omega} 残差 {residual}"
            # 短周期频率高于长周期
            assert omega_s > omega_l

    def test_anchor_is_triangular_point(self, earth_moon_system):
        """锚点是等边三角形顶点 (0.5−μ, ±√3/2, 0)。"""
        mu = earth_moon_system.mu
        _, _, _, _, _, _, x_L4 = _triangular_modes(earth_moon_system, 4)
        _, _, _, _, _, _, x_L5 = _triangular_modes(earth_moon_system, 5)
        np.testing.assert_allclose(x_L4, [0.5 - mu, np.sqrt(3) / 2, 0.0], atol=1e-12)
        np.testing.assert_allclose(x_L5, [0.5 - mu, -np.sqrt(3) / 2, 0.0], atol=1e-12)


class TestInitialGuess:
    def test_l4_l5_are_mirror_images(self, earth_moon_system):
        """L4/L5 初猜关于 x 轴对称（y 分量反号）。"""
        state4, period4 = compute_triangular_initial_guess(
            earth_moon_system, 4, 8000.0, 6000.0, 0.0, 0.0
        )
        state5, period5 = compute_triangular_initial_guess(
            earth_moon_system, 5, 8000.0, 6000.0, 0.0, 0.0
        )
        assert period4 == period5
        np.testing.assert_allclose(state4[0], state5[0])
        np.testing.assert_allclose(state4[1], -state5[1])
        np.testing.assert_allclose(state4[2], state5[2])

    def test_larger_amplitude_gives_larger_offset(self, earth_moon_system):
        """面内振幅翻倍 → 距锚点的位置偏移近似翻倍。"""
        small, _ = compute_triangular_initial_guess(earth_moon_system, 4, 4000.0, 3000.0, 0.0, 0.0)
        large, _ = compute_triangular_initial_guess(earth_moon_system, 4, 8000.0, 6000.0, 0.0, 0.0)
        anchor = earth_moon_system.get_libration_point(LibrationPoint.L4)
        d_small = np.linalg.norm(small[:3] - anchor)
        d_large = np.linalg.norm(large[:3] - anchor)
        assert d_large / d_small == pytest.approx(2.0, rel=0.05)


# =============================================================================
# 扩展测试
# =============================================================================


class TestTriangularInitialGuessExtended:
    """compute_triangular_initial_guess 扩展测试（状态结构、模态、振幅分拆）。"""

    @pytest.mark.parametrize("point", [4, 5])
    def test_state_shape_is_6(self, earth_moon_system, point: int):
        """状态向量形状应为 (6,)。"""
        state0, _ = compute_triangular_initial_guess(
            earth_moon_system, point, 8000.0, 6000.0, 0.0, 0.0
        )
        assert np.asarray(state0).shape == (6,)

    @pytest.mark.parametrize("point", [4, 5])
    def test_period_is_positive(self, earth_moon_system, point: int):
        """标称周期应为正值。"""
        _, T = compute_triangular_initial_guess(earth_moon_system, point, 8000.0, 6000.0, 0.0, 0.0)
        assert T > 0

    @pytest.mark.parametrize("point", [4, 5])
    def test_three_modal_frequencies_present(self, earth_moon_system, point: int):
        """三角点应有三个独立模态频率（短周期、长周期、垂直），两两不等。"""
        omega_s, _, omega_l, _, omega_v, _, _ = _triangular_modes(earth_moon_system, point)
        freqs = sorted([omega_s, omega_l, omega_v])
        # 三个频率应互不相同（特征方程给出两个面内 + 一个面外）
        assert freqs[1] - freqs[0] > 1e-6
        assert freqs[2] - freqs[1] > 1e-6

    @pytest.mark.parametrize("point", [4, 5])
    def test_inplane_amplitude_split_equally(self, earth_moon_system, point: int):
        """面内振幅均分给短周期和长周期模态：两个模态使用相同的 0.5 倍原始振幅。"""
        omega_s, v_s, omega_l, v_l, _, _, _ = _triangular_modes(earth_moon_system, point)
        l_c = earth_moon_system.characteristic_length
        amplitude_in_km = 8000.0
        raw_in = amplitude_in_km / l_c
        # 两个模态的位置振幅贡献：alpha * |v[:3]|，应各为 raw_in 的一半
        pos_amp_s = (0.5 * raw_in / np.linalg.norm(v_s[:3])) * np.linalg.norm(v_s[:3])
        pos_amp_l = (0.5 * raw_in / np.linalg.norm(v_l[:3])) * np.linalg.norm(v_l[:3])
        assert pos_amp_s == pytest.approx(0.5 * raw_in, rel=1e-10)
        assert pos_amp_l == pytest.approx(0.5 * raw_in, rel=1e-10)

    @pytest.mark.parametrize("point", [4, 5])
    def test_larger_amplitude_gives_larger_offset(self, earth_moon_system, point: int):
        """振幅翻倍 → 距锚点的位置偏移近似翻倍（参数化 L4/L5）。"""
        small, _ = compute_triangular_initial_guess(
            earth_moon_system, point, 4000.0, 3000.0, 0.0, 0.0
        )
        large, _ = compute_triangular_initial_guess(
            earth_moon_system, point, 8000.0, 6000.0, 0.0, 0.0
        )
        anchor = earth_moon_system.get_libration_point(LibrationPoint(point))
        d_small = np.linalg.norm(small[:3] - anchor)
        d_large = np.linalg.norm(large[:3] - anchor)
        assert d_large / d_small == pytest.approx(2.0, rel=0.05)
