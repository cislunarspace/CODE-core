"""rk_step 单步行为测试。

三个 RK 方法（PD45/PD78/RK89）共用同一 explicit_rk_step 内核，
单步正确性、非法参数与 state_error_dim 语义按方法参数化验证。
端到端对照（圆轨道、LEO J2 vs DOP853）见 test_rk_family.py。
"""

import math

import numpy as np
import pytest

from e2m2e.integrators import RkMethod, rk_step

pytestmark = pytest.mark.integrator

# (method, h, 单步精度阈值) —— 高阶方法在更大步长上达到更高精度。
HARMONIC_CASES = [
    pytest.param(RkMethod.PD45, 1e-4, 1e-10, id="PD45"),
    pytest.param(RkMethod.PD78, 1e-4, 1e-12, id="PD78"),
    pytest.param(RkMethod.RK89, 1e-3, 1e-13, id="RK89"),
]

RK_METHODS = [
    pytest.param(RkMethod.PD45, id="PD45"),
    pytest.param(RkMethod.PD78, id="PD78"),
    pytest.param(RkMethod.RK89, id="RK89"),
]


def _harmonic(t, y):  # noqa: ARG001
    """y'' = -y 的一阶形式。"""
    return np.array([y[1], -y[0]], dtype=float)


def _two_body(t, y):  # noqa: ARG001
    """归一化二体加速度。"""
    y = np.asarray(y, dtype=float)
    r = y[:3]
    v = y[3:]
    r_norm = np.linalg.norm(r)
    return np.concatenate([v, -r / r_norm**3])


# ---------------------------------------------------------------------------
# 单步正确性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "h", "atol"), HARMONIC_CASES)
def test_harmonic_oscillator_single_step(method, h, atol):
    """单步传播谐波振荡器，与解析解 [cos h, -sin h] 对照。"""
    y0 = np.array([1.0, 0.0], dtype=float)
    result = rk_step(method, 0.0, y0, h, 1e-12, _harmonic)

    expected = np.array([math.cos(h), -math.sin(h)], dtype=float)
    assert np.linalg.norm(np.asarray(result.y_new) - expected) < atol
    assert result.error < atol
    assert result.h_next > 0.0


# ---------------------------------------------------------------------------
# 非法输入
# ---------------------------------------------------------------------------


def test_invalid_step_size_raises():
    """非正步长在积分前被拒绝。"""
    y0 = np.array([1.0, 0.0], dtype=float)
    for h in (0.0, -1e-3):
        with pytest.raises(ValueError):
            rk_step(RkMethod.PD45, 0.0, y0, h, 1e-12, _harmonic)


def test_invalid_tolerance_raises():
    """非正容差在积分前被拒绝。"""
    y0 = np.array([1.0, 0.0], dtype=float)
    with pytest.raises(ValueError):
        rk_step(RkMethod.PD45, 0.0, y0, 1e-3, 0.0, _harmonic)


def test_callback_dimension_mismatch_raises():
    """回调返回维度与状态不符时被拒绝。"""

    def f_bad(t, y):  # noqa: ARG001
        return np.array([0.0], dtype=float)

    y0 = np.array([1.0, 0.0], dtype=float)
    with pytest.raises(ValueError):
        rk_step(RkMethod.PD45, 0.0, y0, 1e-3, 1e-12, f_bad)


# ---------------------------------------------------------------------------
# state_error_dim：STM 增广传播只统计前 N 维误差
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", RK_METHODS)
def test_state_error_dim_default_matches_full(method):
    """state_error_dim=None 时误差与显式全状态维数一致（旧行为）。"""
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
    r_default = rk_step(method, 0.0, y0, 0.01, 1e-10, _two_body)
    r_dim6 = rk_step(method, 0.0, y0, 0.01, 1e-10, _two_body, state_error_dim=6)
    assert abs(r_default.error - r_dim6.error) < 1e-15


@pytest.mark.parametrize("method", RK_METHODS)
def test_state_error_dim_excludes_stm_components(method):
    """42 维增广状态传 state_error_dim=6 时，误差只反映前 6 维物理状态。

    构造一个 42 维系统：前 6 维是二体轨道，后 36 维是 STM（初始单位阵）。
    全状态误差会被 STM 分量主导；分段误差只看前 6 维，与纯 6 维一致。
    """

    def eom_6(t, y):
        return _two_body(t, y)

    def eom_42(t, y):
        y = np.asarray(y, dtype=float)
        state = y[:6]
        stm = y[6:].reshape(6, 6)
        r = state[:3]
        v = state[3:]
        rn = np.linalg.norm(r)
        acc = -r / rn**3
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = -np.eye(3) / rn**3 + 3.0 * np.outer(r, r) / rn**5
        return np.concatenate([v, acc, (A @ stm).flatten()])

    y6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
    y42 = np.concatenate([y6, np.eye(6).flatten()])

    r_6 = rk_step(method, 0.0, y6, 0.01, 1e-10, eom_6)
    r_42_dim6 = rk_step(method, 0.0, y42, 0.01, 1e-10, eom_42, state_error_dim=6)
    r_42_full = rk_step(method, 0.0, y42, 0.01, 1e-10, eom_42)

    # 分段误差 ≈ 纯 6 维误差
    assert abs(r_42_dim6.error - r_6.error) < 1e-12
    # 全状态误差明显更大（STM 分量贡献）
    assert r_42_full.error > r_42_dim6.error


def test_state_error_dim_rejects_invalid():
    """state_error_dim=0 或超过状态长度时抛 ValueError。"""

    def f_zero(t, y):  # noqa: ARG001
        return np.zeros_like(y)

    y0 = np.zeros(6)
    for dim in (0, 10):
        with pytest.raises(ValueError, match="state_error_dim"):
            rk_step(RkMethod.PD45, 0.0, y0, 0.01, 1e-10, f_zero, state_error_dim=dim)
