"""Cowell（Störmer-Cowell）八阶二重积分器测试。

覆盖 J2 归一化加速度、启动历史与位置传播。
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from e2m2e.integrators import (
    RkMethod,
    cowell_step,
    initialize_cowell_history,
)
from tests.integrators.conftest import EARTH_J2, normalized_leo_j2, propagate_rk


def _j2_accel_normalised(t: float, x: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """Two-body + J2 acceleration as a function of position only (normalised:
    mu = re = 1). Mirrors the acceleration slice of conftest.j2_rhs so Cowell
    (which integrates x'' = a(t, x)) sees the same physics DOP853 does."""
    r = np.asarray(x, dtype=float)
    r_norm = np.linalg.norm(r)
    r2 = r_norm**2
    a_2body = -r / r_norm**3
    k = 1.5 * EARTH_J2 / r_norm**5
    z2_over_r2 = r[2] ** 2 / r2
    a_j2 = -k * np.array(
        [
            r[0] * (1.0 - 5.0 * z2_over_r2),
            r[1] * (1.0 - 5.0 * z2_over_r2),
            r[2] * (3.0 - 5.0 * z2_over_r2),
        ]
    )
    return a_2body + a_j2


def _propagate_cowell(accel, x0, v0, h, target_t, t0=0.0, tol=1e-12):
    """Fixed-step Cowell position propagation to ``target_t``. Returns (t, x)."""
    t, x, _v, history = initialize_cowell_history(t0, x0, v0, h, accel, tol=tol)
    n_steps = int(round((target_t - t) / h))
    for _ in range(n_steps):
        result = cowell_step(t, h, tol, accel, history)
        x = np.asarray(result.x_new, dtype=float)
        t += h
        history = result.history
    return t, x


def test_cowell_imports():
    """Cowell is exposed via the public integrators module."""
    assert cowell_step is not None
    assert initialize_cowell_history is not None


def test_cowell_step_history_length_validation():
    """cowell_step rejects a history that is not 10 vectors."""
    accel = lambda t, x: -x  # noqa: E731

    # Cowell needs 10 history samples (2 positions + 8 accelerations); pass 5.
    with pytest.raises(ValueError):
        cowell_step(0.0, 0.1, 1e-12, accel, [[1.0]] * 5)


def test_initialize_cowell_history_fills_ten_samples():
    """7 RK89 startup steps + initial accel yield a 10-vector history."""
    accel = lambda t, x: -x  # noqa: E731
    x0 = np.array([1.0])
    v0 = np.array([0.0])
    h = 0.01

    t, x, v, history = initialize_cowell_history(0.0, x0, v0, h, accel)
    assert len(history) == 10
    assert all(len(sample) == 1 for sample in history)
    assert abs(t - 7 * h) < 1e-12
    # After 7 steps of harmonic motion x ≈ cos(7h).
    assert abs(x[0] - np.cos(7 * h)) < 1e-8


def test_initialize_cowell_history_rejects_short_startup():
    """n_startup < 7 cannot fill 8 acceleration samples."""
    accel = lambda t, x: -x  # noqa: E731
    with pytest.raises(ValueError):
        initialize_cowell_history(0.0, np.array([1.0]), np.array([0.0]), 0.01, accel, n_startup=3)


def test_cowell_harmonic_matches_analytic():
    """Cowell propagates the harmonic oscillator close to the analytic solution."""

    def accel(t, x):  # noqa: ARG001
        return -np.asarray(x, dtype=float)

    x0 = np.array([1.0])
    v0 = np.array([0.0])
    h = 0.01
    t_final = 1.0

    _, x = _propagate_cowell(accel, x0, v0, h, t_final)
    assert abs(x[0] - np.cos(t_final)) < 1e-9


def test_cowell_leo_j2_matches_dop853():
    """Cowell on a fixed step matches scipy DOP853 over ~1 day (< 1e-9 position).

    Cowell is position-only and fixed-step, so it lands on the nearest h-multiple
    of the target; we compare its position against DOP853's dense output at the
    exact t Cowell reached.
    """
    rhs, y0, t_span = normalized_leo_j2(days=1.0)
    x0, v0 = y0[:3].copy(), y0[3:].copy()
    accel = _j2_accel_normalised

    h = 0.04  # normalised time units (~32 s for a 400 km LEO)
    t_cw, x_cw = _propagate_cowell(accel, x0, v0, h, t_span[1])

    sol = solve_ivp(
        rhs, t_span, y0, method="DOP853", rtol=1e-12, atol=1e-12, dense_output=True
    )
    assert sol.success
    x_ref = np.asarray(sol.sol(t_cw))[:3]
    assert np.linalg.norm(x_cw - x_ref) < 1e-9


def test_cowell_position_matches_rk89():
    """Cross-check: Cowell (fixed-step position) agrees with RK89 (adaptive
    full-state) on the J2 problem at the same final time.

    Both are compared at ``t_cw`` (Cowell's fixed-step landing point), so the
    satellite has not moved between the two evaluations. Illustrative
    cross-integrator agreement; the DOP853 comparison above is the primary gate.
    """
    rhs, y0, t_span = normalized_leo_j2(days=1.0)
    x0, v0 = y0[:3].copy(), y0[3:].copy()
    accel = _j2_accel_normalised

    h = 0.04
    t_cw, x_cw = _propagate_cowell(accel, x0, v0, h, t_span[1])

    # Adaptive RK89 reference to the SAME time Cowell landed on.
    _, y_rk, _ = propagate_rk(RkMethod.RK89, rhs, y0, (float(t_span[0]), t_cw), tol=1e-12)
    assert np.linalg.norm(x_cw - y_rk[:3]) < 1e-8
