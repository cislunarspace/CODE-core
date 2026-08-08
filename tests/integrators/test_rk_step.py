"""Rust 扩展积分器模块测试。

覆盖模块导入、RK 步进符号与步长非正校验。
"""

import math

import numpy as np
import pytest

pytestmark = [pytest.mark.l1]


def test_hello_integrators_smoke():
    """Smoke test: the Rust extension module imports and responds."""
    from e2m2e.integrators import hello_integrators

    assert hello_integrators() == "hello from e2m2e-integrators"


def test_rk_step_imports():
    """The RK stepper symbols are importable from the extension module."""
    from e2m2e.integrators import RkMethod, rk_step

    assert RkMethod.PD45 is not None
    assert callable(rk_step)


def test_rk_step_harmonic_oscillator():
    """PD45 approximates the harmonic oscillator on a small step."""
    from e2m2e.integrators import RkMethod, rk_step

    def f(t, y):
        return np.array([y[1], -y[0]], dtype=float)

    y0 = np.array([1.0, 0.0], dtype=float)
    h = 1e-4
    result = rk_step(RkMethod.PD45, 0.0, y0, h, 1e-12, f)

    expected = np.array([math.cos(h), -math.sin(h)], dtype=float)
    assert np.linalg.norm(np.asarray(result.y_new) - expected) < 1e-10
    assert result.error < 1e-10
    assert result.h_next > 0.0


def test_rk_step_invalid_step_size_raises():
    """A non-positive step size is rejected before integration."""
    from e2m2e.integrators import RkMethod, rk_step

    def f(t, y):
        return np.zeros_like(y)

    y0 = np.array([1.0, 0.0], dtype=float)

    with pytest.raises(ValueError):
        rk_step(RkMethod.PD45, 0.0, y0, 0.0, 1e-12, f)

    with pytest.raises(ValueError):
        rk_step(RkMethod.PD45, 0.0, y0, -1e-3, 1e-12, f)


def test_rk_step_invalid_tolerance_raises():
    """A non-positive tolerance is rejected before integration."""
    from e2m2e.integrators import RkMethod, rk_step

    def f(t, y):
        return np.zeros_like(y)

    y0 = np.array([1.0, 0.0], dtype=float)

    with pytest.raises(ValueError):
        rk_step(RkMethod.PD45, 0.0, y0, 1e-3, 0.0, f)


def test_rk_step_callback_dimension_mismatch_raises():
    """A callback returning the wrong dimension is rejected."""
    from e2m2e.integrators import RkMethod, rk_step

    def f(t, y):
        return np.array([0.0], dtype=float)

    y0 = np.array([1.0, 0.0], dtype=float)

    with pytest.raises(ValueError):
        rk_step(RkMethod.PD45, 0.0, y0, 1e-3, 1e-12, f)


def test_public_shim_imports():
    """The recommended public entry point re-exports the stepper symbols."""
    from e2m2e.integrators import RkMethod, rk_step

    assert callable(rk_step)
    assert RkMethod.PD45 is not None


def test_two_body_circular_orbit_matches_scipy():
    """PD45 follows a circular two-body orbit as closely as scipy RK45."""
    from scipy.integrate import solve_ivp

    from e2m2e.integrators import RkMethod, rk_step

    def two_body(t, y):
        r = y[:3]
        v = y[3:]
        r_norm = np.linalg.norm(r)
        a = -r / r_norm**3
        return np.concatenate([v, a])

    # Unit circle in xy-plane, circular velocity for mu=1
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
    period = 2.0 * np.pi
    tol = 1e-12

    # Propagate with Rust PD45 using adaptive step acceptance
    t = 0.0
    y = y0.copy()
    h = 0.01
    while t < period:
        h = min(h, period - t)
        result = rk_step(RkMethod.PD45, t, y, h, tol, two_body)
        y = np.asarray(result.y_new)
        t += h
        h = result.h_next

    # Propagate with scipy RK45
    sol = solve_ivp(two_body, (0.0, period), y0, method="RK45", rtol=tol, atol=tol)
    y_scipy = sol.y[:, -1]

    assert np.linalg.norm(y - y_scipy) < 1e-9


def test_two_body_circular_orbit_matches_analytic():
    """PD45 follows a circular two-body orbit to high analytic precision."""
    from e2m2e.integrators import RkMethod, rk_step
    from tests.integrators.conftest import kepler_analytic_state

    def two_body(t, y):
        r = y[:3]
        v = y[3:]
        r_norm = np.linalg.norm(r)
        a = -r / r_norm**3
        return np.concatenate([v, a])

    y0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
    r0 = y0[:3]
    v0 = y0[3:]
    period = 2.0 * np.pi
    tol = 1e-12

    t = 0.0
    y = y0.copy()
    h = 0.01
    while t < period:
        h = min(h, period - t)
        result = rk_step(RkMethod.PD45, t, y, h, tol, two_body)
        y = np.asarray(result.y_new)
        t += h
        h = result.h_next

    y_exact = kepler_analytic_state(r0, v0, period)
    assert np.linalg.norm(y - y_exact) < 1e-9


def test_rk_step_state_error_dim_default_matches_full():
    """state_error_dim=None 时误差与旧行为一致（全状态 L2）。"""
    from e2m2e.integrators import RkMethod, rk_step

    def f(t, y):
        y = np.asarray(y, dtype=float)
        r = y[:3]
        v = y[3:]
        r_norm = np.linalg.norm(r)
        return np.concatenate([v, -r / r_norm**3])

    y0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
    r_default = rk_step(RkMethod.PD45, 0.0, y0, 0.01, 1e-10, f)
    r_dim6 = rk_step(RkMethod.PD45, 0.0, y0, 0.01, 1e-10, f, state_error_dim=6)
    assert abs(r_default.error - r_dim6.error) < 1e-15


def test_rk_step_state_error_dim_excludes_stm_components():
    """42 维增广状态传 state_error_dim=6 时，误差只反映前 6 维物理状态。

    构造一个 42 维系统：前 6 维是二体轨道，后 36 维是 STM（初始单位阵）。
    全状态误差会被 STM 分量主导；分段误差只看前 6 维，与纯 6 维一致。
    """
    from e2m2e.integrators import RkMethod, rk_step

    def eom_6(t, y):
        y = np.asarray(y, dtype=float)
        r = y[:3]
        v = y[3:]
        rn = np.linalg.norm(r)
        return np.concatenate([v, -r / rn**3])

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

    r_6 = rk_step(RkMethod.PD45, 0.0, y6, 0.01, 1e-10, eom_6)
    r_42_dim6 = rk_step(RkMethod.PD45, 0.0, y42, 0.01, 1e-10, eom_42, state_error_dim=6)
    r_42_full = rk_step(RkMethod.PD45, 0.0, y42, 0.01, 1e-10, eom_42)

    # 分段误差 ≈ 纯 6 维误差
    assert abs(r_42_dim6.error - r_6.error) < 1e-12
    # 全状态误差明显更大（STM 分量贡献）
    assert r_42_full.error > r_42_dim6.error


def test_rk_step_state_error_dim_rejects_invalid():
    """state_error_dim=0 或超过状态长度时抛 ValueError。"""
    from e2m2e.integrators import RkMethod, rk_step

    def f(t, y):
        return np.zeros_like(y)

    y0 = np.zeros(6)
    with pytest.raises(ValueError, match="state_error_dim"):
        rk_step(RkMethod.PD45, 0.0, y0, 0.01, 1e-10, f, state_error_dim=0)
    with pytest.raises(ValueError, match="state_error_dim"):
        rk_step(RkMethod.PD45, 0.0, y0, 0.01, 1e-10, f, state_error_dim=10)
