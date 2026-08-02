"""Lissajous 初始猜测模块测试。

覆盖 compute_lissajous_initial_guess 的返回结构、锚点约束、振幅独立性、
相位影响与错误处理。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_System, LibrationPoint
from e2m2e.algorithm.family.lissajous_initial_guess import compute_lissajous_initial_guess

MU_EM = 1.215058560962404e-2


def _earth_moon_system() -> CR3BP_System:
    system = CR3BP_System(mu=MU_EM, primary="Earth", secondary="Moon")
    system.set_characteristic_scales(distance=384405.0, period=27.32 * 86400.0)
    return system


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

    def test_returns_state_and_period(self):
        """应返回 (state, period) 元组。"""
        system = _earth_moon_system()
        result = compute_lissajous_initial_guess(system, 2, 1000.0, 5000.0, 0.0, 0.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_state_shape_is_6(self):
        """状态向量形状应为 (6,)。"""
        system = _earth_moon_system()
        state0, _ = compute_lissajous_initial_guess(system, 2, 1000.0, 5000.0, 0.0, 0.0)
        assert np.asarray(state0).shape == (6,)

    def test_period_is_positive(self):
        """标称周期应为正值。"""
        system = _earth_moon_system()
        _, T = compute_lissajous_initial_guess(system, 2, 1000.0, 5000.0, 0.0, 0.0)
        assert T > 0

    @pytest.mark.parametrize("L", [1, 2, 3])
    def test_anchor_near_libration_point(self, L: int):
        """初始状态位置应在对应平动点附近（偏移 < 0.2 DU）。"""
        system = _earth_moon_system()
        state0, _ = compute_lissajous_initial_guess(system, L, 1000.0, 5000.0, 0.0, 0.0)
        lp = system.get_libration_point(LibrationPoint(L))
        # 位置分量偏移应远小于 1 DU（振幅 ~ 数千 km / 384405 km << 1）
        dist = np.linalg.norm(state0[:3] - lp)
        assert dist < 0.2

    @pytest.mark.parametrize("L", [1, 2])
    def test_larger_amplitude_gives_larger_offset(self, L: int):
        """面内振幅翻倍 → 距锚点的位置偏移近似翻倍。"""
        system = _earth_moon_system()
        small, _ = compute_lissajous_initial_guess(system, L, 1000.0, 2000.0, 0.0, 0.0)
        large, _ = compute_lissajous_initial_guess(system, L, 2000.0, 2000.0, 0.0, 0.0)
        lp = system.get_libration_point(LibrationPoint(L))
        d_small = np.linalg.norm(small[:3] - lp)
        d_large = np.linalg.norm(large[:3] - lp)
        assert d_large > d_small

    @pytest.mark.parametrize("L", [1, 2])
    def test_inplane_amplitude_affects_position(self, L: int):
        """改变面内振幅应改变初始位置。"""
        system = _earth_moon_system()
        state_a, _ = compute_lissajous_initial_guess(system, L, 1000.0, 5000.0, 0.0, 0.0)
        state_b, _ = compute_lissajous_initial_guess(system, L, 3000.0, 5000.0, 0.0, 0.0)
        assert not np.allclose(state_a, state_b)

    @pytest.mark.parametrize("L", [1, 2])
    def test_outplane_amplitude_affects_position(self, L: int):
        """改变面外振幅应改变初始位置。"""
        system = _earth_moon_system()
        state_a, _ = compute_lissajous_initial_guess(system, L, 1000.0, 2000.0, 0.0, 0.0)
        state_b, _ = compute_lissajous_initial_guess(system, L, 1000.0, 8000.0, 0.0, 0.0)
        assert not np.allclose(state_a, state_b)

    def test_invalid_collinear_point_raises(self):
        """collinear_point 非 1/2/3 时应抛出 ValueError。"""
        system = _earth_moon_system()
        with pytest.raises(ValueError, match="collinear_point"):
            compute_lissajous_initial_guess(system, 4, 1000.0, 5000.0, 0.0, 0.0)

    @pytest.mark.parametrize("L", [1, 2, 3])
    def test_phase_affects_initial_state(self, L: int):
        """改变面内相位应改变初始状态。"""
        system = _earth_moon_system()
        state_0, _ = compute_lissajous_initial_guess(system, L, 1000.0, 5000.0, 0.0, 0.0)
        state_1, _ = compute_lissajous_initial_guess(system, L, 1000.0, 5000.0, 0.5, 0.0)
        assert not np.allclose(state_0, state_1)

    @pytest.mark.parametrize("L", [1, 2])
    def test_period_reasonable_range(self, L: int):
        """L1/L2 面内周期应在 Lyapunov 量级（约 2-4 无量纲时间）。"""
        system = _earth_moon_system()
        _, T = compute_lissajous_initial_guess(system, L, 1000.0, 5000.0, 0.0, 0.0)
        assert 1.5 < T < 5.0

    @pytest.mark.parametrize("L", [1, 2, 3])
    def test_state_finite(self, L: int):
        """状态所有分量应为有限值。"""
        system = _earth_moon_system()
        state0, T = compute_lissajous_initial_guess(system, L, 1000.0, 5000.0, 0.0, 0.0)
        assert np.all(np.isfinite(state0))
        assert np.isfinite(T)
