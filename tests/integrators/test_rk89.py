"""RK89（Verner 9(8)）积分器测试。

覆盖谐振子、圆轨道二体解析精度。
"""

import math

import numpy as np
from scipy.integrate import solve_ivp

from tests.integrators.conftest import (
    kepler_analytic_state,
    normalized_leo_j2,
    propagate_rk,
)


def test_rk89_imports():
    """RK89 is selectable via the public RkMethod enum."""
    from e2m2e.integrators import RkMethod

    assert RkMethod.RK89 is not None


def test_rk89_harmonic_oscillator():
    """RK89 on a small step is far more accurate than its tolerance band."""
    from e2m2e.integrators import RkMethod, rk_step

    def f(t, y):  # noqa: ARG001
        return np.array([y[1], -y[0]], dtype=float)

    y0 = np.array([1.0, 0.0], dtype=float)
    h = 1e-3
    result = rk_step(RkMethod.RK89, 0.0, y0, h, 1e-12, f)

    expected = np.array([math.cos(h), -math.sin(h)], dtype=float)
    assert np.linalg.norm(np.asarray(result.y_new) - expected) < 1e-13
    assert result.error < 1e-13
    assert result.h_next > 0.0


def test_rk89_two_body_circular_matches_analytic():
    """RK89 propagates a circular two-body orbit to high analytic precision."""
    from e2m2e.integrators import RkMethod

    def two_body(t, y):  # noqa: ARG001
        r = y[:3]
        v = y[3:]
        r_norm = np.linalg.norm(r)
        a = -r / r_norm**3
        return np.concatenate([v, a])

    y0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
    period = 2.0 * np.pi
    _, y_final, _ = propagate_rk(RkMethod.RK89, two_body, y0, (0.0, period), tol=1e-13, h0=0.01)

    y_exact = kepler_analytic_state(y0[:3], y0[3:], period)
    assert np.linalg.norm(y_final - y_exact) < 1e-11


def test_rk89_leo_j2_matches_dop853():
    """RK89 normalised-LEO + J2 propagation over 1 day matches scipy DOP853."""
    from e2m2e.integrators import RkMethod

    rhs, y0, t_span = normalized_leo_j2(days=1.0)
    _, y_rk89, _ = propagate_rk(RkMethod.RK89, rhs, y0, t_span, tol=1e-13, h0=0.01)

    sol = solve_ivp(rhs, t_span, y0, method="DOP853", rtol=1e-12, atol=1e-12)
    assert sol.success
    assert np.linalg.norm(y_rk89 - sol.y[:, -1]) < 1e-9
