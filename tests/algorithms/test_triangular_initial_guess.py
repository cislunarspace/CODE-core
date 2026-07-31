"""L4/L5 三角平动点初猜测试（纯 CR3BP，不依赖 SPICE 内核）。"""

import numpy as np
import pytest

from e2m2e.algorithms.triangular_initial_guess import (
    _triangular_modes,
    compute_triangular_initial_guess,
)
from e2m2e.core import CR3BP_System, LibrationPoint


def _earth_moon_system():
    system = CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()
    system.set_characteristic_scales(384400.0, 27.32 * 86400.0)
    system.compute_libration_points()
    return system


class TestTriangularModes:
    def test_frequencies_satisfy_characteristic_equation(self):
        """纯 CR3BP 下 ω_s/ω_l 满足 ω⁴ − ω² + 27μ(1−μ)/4 = 0。"""
        system = _earth_moon_system()
        mu = system.mu
        for point in (4, 5):
            omega_s, _, omega_l, _, _omega_v, _, x_L = _triangular_modes(system, point)
            for omega in (omega_s, omega_l):
                residual = omega**4 - omega**2 + 27.0 * mu * (1.0 - mu) / 4.0
                assert abs(residual) < 1e-6, f"L{point} ω={omega} 残差 {residual}"
            # 短周期频率高于长周期
            assert omega_s > omega_l

    def test_anchor_is_triangular_point(self):
        """锚点是等边三角形顶点 (0.5−μ, ±√3/2, 0)。"""
        system = _earth_moon_system()
        mu = system.mu
        _, _, _, _, _, _, x_L4 = _triangular_modes(system, 4)
        _, _, _, _, _, _, x_L5 = _triangular_modes(system, 5)
        np.testing.assert_allclose(x_L4, [0.5 - mu, np.sqrt(3) / 2, 0.0], atol=1e-12)
        np.testing.assert_allclose(x_L5, [0.5 - mu, -np.sqrt(3) / 2, 0.0], atol=1e-12)


class TestInitialGuess:
    def test_l4_l5_are_mirror_images(self):
        """L4/L5 初猜关于 x 轴对称（y 分量反号）。"""
        system = _earth_moon_system()
        state4, period4 = compute_triangular_initial_guess(
            system, 4, 8000.0, 6000.0, 0.0, 0.0
        )
        state5, period5 = compute_triangular_initial_guess(
            system, 5, 8000.0, 6000.0, 0.0, 0.0
        )
        assert period4 == period5
        np.testing.assert_allclose(state4[0], state5[0])
        np.testing.assert_allclose(state4[1], -state5[1])
        np.testing.assert_allclose(state4[2], state5[2])

    def test_larger_amplitude_gives_larger_offset(self):
        """面内振幅翻倍 → 距锚点的位置偏移近似翻倍。"""
        system = _earth_moon_system()
        small, _ = compute_triangular_initial_guess(system, 4, 4000.0, 3000.0, 0.0, 0.0)
        large, _ = compute_triangular_initial_guess(system, 4, 8000.0, 6000.0, 0.0, 0.0)
        anchor = system.get_libration_point(LibrationPoint.L4)
        d_small = np.linalg.norm(small[:3] - anchor)
        d_large = np.linalg.norm(large[:3] - anchor)
        assert d_large / d_small == pytest.approx(2.0, rel=0.05)
