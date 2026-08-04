"""Axial 轨道初始猜测模块测试。

覆盖 compute_axial_initial_guess 的返回结构、x 轴对称初始条件、
面外速度非零、分岔振幅与错误处理。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System, LibrationPoint
from e2m2e.algorithm.family.axial_initial_guess import compute_axial_initial_guess

MU_EM = 1.215058560962404e-2


def _earth_moon_dynamics() -> CR3BP_Dynamics:
    system = CR3BP_System(mu=MU_EM, primary="Earth", secondary="Moon")
    system.set_characteristic_scales(distance=384405.0, period=27.32 * 86400.0)
    return CR3BP_Dynamics(system)


# =============================================================================
# Module identity
# =============================================================================


class TestModuleImport:
    """验证模块可直接导入，函数可直接调用。"""

    def test_module_importable(self):
        """axial_initial_guess 模块应可直接导入。"""
        assert compute_axial_initial_guess is not None

    def test_function_has_docstring(self):
        """公共函数应有文档字符串。"""
        assert compute_axial_initial_guess.__doc__ is not None


# =============================================================================
# compute_axial_initial_guess
# =============================================================================


class TestAxialInitialGuess:
    """compute_axial_initial_guess 的正确性。"""

    def test_returns_state_and_period(self):
        """应返回 (state, period) 元组。"""
        dynamics = _earth_moon_dynamics()
        result = compute_axial_initial_guess(dynamics, 1, 0.01)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_state_shape_is_6(self):
        """状态向量形状应为 (6,)。"""
        dynamics = _earth_moon_dynamics()
        state0, _ = compute_axial_initial_guess(dynamics, 1, 0.01)
        assert np.asarray(state0).shape == (6,)

    def test_period_is_positive(self):
        """标称周期应为正值。"""
        dynamics = _earth_moon_dynamics()
        _, T = compute_axial_initial_guess(dynamics, 1, 0.01)
        assert T > 0

    def test_z0_is_zero(self):
        """初始 z 坐标应为 0（x 轴对称，Type B）。"""
        dynamics = _earth_moon_dynamics()
        state0, _ = compute_axial_initial_guess(dynamics, 1, 0.01)
        assert state0[2] == pytest.approx(0.0, abs=1e-15)

    def test_y0_is_zero(self):
        """初始 y 坐标应为 0（x 轴上）。"""
        dynamics = _earth_moon_dynamics()
        state0, _ = compute_axial_initial_guess(dynamics, 1, 0.01)
        assert state0[1] == pytest.approx(0.0, abs=1e-15)

    def test_xdot0_is_zero(self):
        """初始 x 方向速度应为 0（x 轴垂直穿越）。"""
        dynamics = _earth_moon_dynamics()
        state0, _ = compute_axial_initial_guess(dynamics, 1, 0.01)
        assert state0[3] == pytest.approx(0.0, abs=1e-15)

    def test_vz0_nonzero(self):
        """初始 z 方向速度应非零（Type B 特征）。"""
        dynamics = _earth_moon_dynamics()
        state0, _ = compute_axial_initial_guess(dynamics, 1, 0.01)
        assert abs(state0[5]) > 1e-10

    def test_vz0_sign_follows_input(self):
        """vz0 符号应与输入一致（正上族、负下族）。"""
        dynamics = _earth_moon_dynamics()
        state_pos, _ = compute_axial_initial_guess(dynamics, 1, 0.01)
        state_neg, _ = compute_axial_initial_guess(dynamics, 1, -0.01)
        assert state_pos[5] > 0
        assert state_neg[5] < 0

    def test_anchor_has_bifurcation_amplitude(self):
        """初始 x 坐标应远离平动点（继承 Lyapunov 垂直临界轨道面内振幅）。

        真 Axial 分岔点的 x₀ 距 L1 约 0.055 DU（~21000 km），
        而非紧邻 L1（那是 Vertical Lyapunov 种子的特征）。
        """
        dynamics = _earth_moon_dynamics()
        state0, _ = compute_axial_initial_guess(dynamics, 1, 0.01)
        lp = dynamics.system.get_libration_point(LibrationPoint.L1)
        assert abs(state0[0] - lp[0]) > 0.02

    def test_lyapunov_inplane_velocity_nonzero(self):
        """初始 y 方向速度应非零（继承 Lyapunov 父支面内振幅）。

        这区分了真 Axial 种子（vy0 ≠ 0）与 Vertical Lyapunov 种子（vy0 = 0）。
        """
        dynamics = _earth_moon_dynamics()
        state0, _ = compute_axial_initial_guess(dynamics, 1, 0.01)
        assert abs(state0[4]) > 0.01

    def test_state_finite(self):
        """状态所有分量应为有限值。"""
        dynamics = _earth_moon_dynamics()
        state0, T = compute_axial_initial_guess(dynamics, 1, 0.01)
        assert np.all(np.isfinite(state0))
        assert np.isfinite(T)

    def test_period_reasonable_range(self):
        """L1 分岔 Lyapunov 周期应在合理量级（约 3-5 无量纲时间）。"""
        dynamics = _earth_moon_dynamics()
        _, T = compute_axial_initial_guess(dynamics, 1, 0.01)
        assert 2.5 < T < 6.0
