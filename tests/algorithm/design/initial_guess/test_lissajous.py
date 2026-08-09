"""Lissajous 初始猜测模块测试。

覆盖 compute_lissajous_initial_guess 的返回结构、锚点约束、振幅独立性、
相位影响与错误处理。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import LibrationPoint
from e2m2e.algorithm.family.lissajous_initial_guess import (
    compute_lissajous_bounded_trajectory,
    compute_lissajous_initial_guess,
)

# =============================================================================
# Module identity
# =============================================================================


class TestModuleImport:
    """验证模块可直接导入，函数可直接调用。"""

    def test_module_importable(self):
        """lissajous_initial_guess 模块应可直接导入。"""
        assert compute_lissajous_initial_guess is not None

    def test_functions_have_docstrings(self):
        """公共函数应有文档字符串。"""
        assert compute_lissajous_initial_guess.__doc__ is not None


# =============================================================================
# compute_lissajous_initial_guess
# =============================================================================


class TestLissajousInitialGuess:
    """compute_lissajous_initial_guess 的正确性。"""

    def test_returns_state_and_period(self, earth_moon_system):
        """应返回 (state, period) 元组。"""
        result = compute_lissajous_initial_guess(earth_moon_system, 2, 1000.0, 5000.0, 0.0, 0.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_state_shape_is_6(self, earth_moon_system):
        """状态向量形状应为 (6,)。"""
        state0, _ = compute_lissajous_initial_guess(earth_moon_system, 2, 1000.0, 5000.0, 0.0, 0.0)
        assert np.asarray(state0).shape == (6,)

    def test_period_is_positive(self, earth_moon_system):
        """标称周期应为正值。"""
        _, T = compute_lissajous_initial_guess(earth_moon_system, 2, 1000.0, 5000.0, 0.0, 0.0)
        assert T > 0

    @pytest.mark.parametrize("L", [1, 2, 3])
    def test_anchor_near_libration_point(self, earth_moon_system, L: int):
        """初始状态位置应在对应平动点附近（偏移 < 0.2 DU）。"""
        state0, _ = compute_lissajous_initial_guess(earth_moon_system, L, 1000.0, 5000.0, 0.0, 0.0)
        lp = earth_moon_system.get_libration_point(LibrationPoint(L))
        # 位置分量偏移应远小于 1 DU（振幅 ~ 数千 km / 384405 km << 1）
        dist = np.linalg.norm(state0[:3] - lp)
        assert dist < 0.2

    @pytest.mark.parametrize("L", [1, 2])
    def test_larger_amplitude_gives_larger_offset(self, earth_moon_system, L: int):
        """面内振幅翻倍 → 距锚点的位置偏移近似翻倍。"""
        small, _ = compute_lissajous_initial_guess(earth_moon_system, L, 1000.0, 2000.0, 0.0, 0.0)
        large, _ = compute_lissajous_initial_guess(earth_moon_system, L, 2000.0, 2000.0, 0.0, 0.0)
        lp = earth_moon_system.get_libration_point(LibrationPoint(L))
        d_small = np.linalg.norm(small[:3] - lp)
        d_large = np.linalg.norm(large[:3] - lp)
        assert d_large > d_small

    @pytest.mark.parametrize("L", [1, 2])
    def test_inplane_amplitude_affects_position(self, earth_moon_system, L: int):
        """改变面内振幅应改变初始位置。"""
        state_a, _ = compute_lissajous_initial_guess(earth_moon_system, L, 1000.0, 5000.0, 0.0, 0.0)
        state_b, _ = compute_lissajous_initial_guess(earth_moon_system, L, 3000.0, 5000.0, 0.0, 0.0)
        assert not np.allclose(state_a, state_b)

    @pytest.mark.parametrize("L", [1, 2])
    def test_outplane_amplitude_affects_position(self, earth_moon_system, L: int):
        """改变面外振幅应改变初始位置。"""
        state_a, _ = compute_lissajous_initial_guess(earth_moon_system, L, 1000.0, 2000.0, 0.0, 0.0)
        state_b, _ = compute_lissajous_initial_guess(earth_moon_system, L, 1000.0, 8000.0, 0.0, 0.0)
        assert not np.allclose(state_a, state_b)

    def test_invalid_collinear_point_raises(self, earth_moon_system):
        """collinear_point 非 1/2/3 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="collinear_point"):
            compute_lissajous_initial_guess(earth_moon_system, 4, 1000.0, 5000.0, 0.0, 0.0)

    @pytest.mark.parametrize("L", [1, 2, 3])
    def test_phase_affects_initial_state(self, earth_moon_system, L: int):
        """改变面内相位应改变初始状态。"""
        state_0, _ = compute_lissajous_initial_guess(earth_moon_system, L, 1000.0, 5000.0, 0.0, 0.0)
        state_1, _ = compute_lissajous_initial_guess(earth_moon_system, L, 1000.0, 5000.0, 0.5, 0.0)
        assert not np.allclose(state_0, state_1)

    @pytest.mark.parametrize("L", [1, 2])
    def test_period_reasonable_range(self, earth_moon_system, L: int):
        """L1/L2 面内周期应在 Lyapunov 量级（约 2-4 无量纲时间）。"""
        _, T = compute_lissajous_initial_guess(earth_moon_system, L, 1000.0, 5000.0, 0.0, 0.0)
        assert 1.5 < T < 5.0

    @pytest.mark.parametrize("L", [1, 2, 3])
    def test_state_finite(self, earth_moon_system, L: int):
        """状态所有分量应为有限值。"""
        state0, T = compute_lissajous_initial_guess(earth_moon_system, L, 1000.0, 5000.0, 0.0, 0.0)
        assert np.all(np.isfinite(state0))
        assert np.isfinite(T)


# =============================================================================
# compute_lissajous_bounded_trajectory
# =============================================================================


@pytest.mark.slow
class TestLissajousBoundedTrajectory:
    """compute_lissajous_bounded_trajectory 返回中心流形约化的多点有界轨迹。"""

    @pytest.mark.parametrize("L", [1, 2])
    @pytest.mark.parametrize("ain, aout", [(500.0, 2000.0), (2500.0, 7500.0)])
    def test_bounded_trajectory(self, earth_moon_system, L: int, ain: float, aout: float):
        """返回多点 synodic 质心系有界轨迹，面内偏移 ~2× 振幅量级。"""
        result = compute_lissajous_bounded_trajectory(earth_moon_system, L, ain, aout, 0.01, 0.55)
        # 三元组 (states, times, period)
        assert isinstance(result, tuple) and len(result) == 3
        states, times, period = result
        states = np.asarray(states)
        times = np.asarray(times)

        # states 形状 (M, 6) 且 M > 1（多点轨迹，非线性种子单点）
        assert states.ndim == 2
        assert states.shape[1] == 6
        assert states.shape[0] > 1

        # times 形状 (M,)、首元素≈0、非负且单调递增
        assert times.shape == (states.shape[0],)
        assert times[0] == pytest.approx(0.0, abs=1e-9)
        assert np.all(times >= 0.0)
        assert np.all(np.diff(times) > 0.0)

        # period > 0
        assert period > 0

        # 有界性：面内偏移相对平动点（km），量级 ~2× 振幅，留 3× 余量
        l_c = earth_moon_system.characteristic_length
        assert l_c is not None
        lp = earth_moon_system.get_libration_point(LibrationPoint(L))
        rel = (states[:, :3] - lp) * l_c
        assert np.max(np.abs(rel[:, :2])) < 3 * ain
