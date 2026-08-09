"""PD78（Dormand-Prince 8(7)13M）积分器测试。

覆盖谐振子、圆轨道二体解析精度。
"""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tests.numerical.integrators.conftest import (
    kepler_analytic_state,
    normalized_leo_j2,
    propagate_rk,
)

pytestmark = [pytest.mark.l1]


def test_pd78_imports():
    """PD78 is selectable via the public RkMethod enum."""
    from e2m2e.integrators import RkMethod

    assert RkMethod.PD78 is not None


def test_pd78_harmonic_oscillator():
    """PD78 on a small step is far more accurate than its tolerance band."""
    from e2m2e.integrators import RkMethod, rk_step

    def f(t, y):  # noqa: ARG001
        return np.array([y[1], -y[0]], dtype=float)

    y0 = np.array([1.0, 0.0], dtype=float)
    h = 1e-4
    result = rk_step(RkMethod.PD78, 0.0, y0, h, 1e-12, f)

    expected = np.array([math.cos(h), -math.sin(h)], dtype=float)
    assert np.linalg.norm(np.asarray(result.y_new) - expected) < 1e-12
    assert result.error < 1e-12
    assert result.h_next > 0.0


def test_pd78_two_body_circular_matches_analytic():
    """PD78 propagates a circular two-body orbit to high analytic precision."""
    from e2m2e.integrators import RkMethod

    def two_body(t, y):  # noqa: ARG001
        r = y[:3]
        v = y[3:]
        r_norm = np.linalg.norm(r)
        a = -r / r_norm**3
        return np.concatenate([v, a])

    y0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
    period = 2.0 * np.pi
    _, y_final, _ = propagate_rk(RkMethod.PD78, two_body, y0, (0.0, period), tol=1e-12, h0=0.01)

    y_exact = kepler_analytic_state(y0[:3], y0[3:], period)
    assert np.linalg.norm(y_final - y_exact) < 1e-10


def test_pd78_leo_j2_matches_dop853():
    """PD78 normalised-LEO + J2 propagation over 1 day matches scipy DOP853.

    Normalised units (DU = Earth radius, TU s.t. mu = 1) keep ||y|| ~ O(1) so
    the relative tolerance in propagate_rk maps to absolute error directly.
    """
    from e2m2e.integrators import RkMethod

    rhs, y0, t_span = normalized_leo_j2(days=1.0)
    _, y_pd78, _ = propagate_rk(RkMethod.PD78, rhs, y0, t_span, tol=1e-12, h0=0.01)

    sol = solve_ivp(rhs, t_span, y0, method="DOP853", rtol=1e-12, atol=1e-12)
    assert sol.success
    assert np.linalg.norm(y_pd78 - sol.y[:, -1]) < 1e-9
