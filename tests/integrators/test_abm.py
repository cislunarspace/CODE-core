"""ABM（Adams-Bashforth-Moulton）多步积分器测试。

覆盖历史长度校验、启动填充、谐振子精度与四阶收敛。
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from e2m2e.integrators import (
    MultistepMethod,
    initialize_abm_history,
    multistep_step,
)
from tests.integrators.conftest import normalized_leo_j2

pytestmark = [pytest.mark.l1]


def _propagate_abm(f, y0, h, target_t, t0=0.0):
    """Fixed-step ABM propagation to ``target_t``. Returns (t, y)."""
    t, y, history = initialize_abm_history(t0, y0, h, f, n_stages=3)
    n_steps = int(round((target_t - t) / h))
    for _ in range(n_steps):
        result = multistep_step(MultistepMethod.ABM, t, y, h, 1e-12, f, history)
        y = np.asarray(result.y_new, dtype=float)
        t += h
        history = result.history
    return t, y


def test_abm_imports():
    """ABM is selectable via the public MultistepMethod enum."""
    assert MultistepMethod.ABM is not None


def test_multistep_step_history_length_validation():
    """multistep_step rejects history of the wrong length."""
    y0 = np.array([1.0, 0.0])

    def f(t, y):  # noqa: ARG001
        return np.array([y[1], -y[0]])

    # ABM needs 4 history samples; pass 3 → error.
    with pytest.raises(ValueError):
        multistep_step(MultistepMethod.ABM, 0.0, y0, 0.1, 1e-12, f, [[0, -1]] * 3)


def test_initialize_abm_history_fills_four_samples():
    """3 RK89 startup steps + initial f yields a 4-sample history."""
    y0 = np.array([1.0, 0.0])

    def f(t, y):  # noqa: ARG001
        return np.array([y[1], -y[0]])

    h = 0.01
    t, y, history = initialize_abm_history(0.0, y0, h, f, n_stages=3)
    assert len(history) == 4
    assert all(len(sample) == 2 for sample in history)
    assert abs(t - 3 * h) < 1e-12
    # After 3 steps of harmonic motion y ≈ [cos(3h), -sin(3h)].
    assert abs(y[0] - np.cos(3 * h)) < 1e-6


def test_abm_harmonic_matches_analytic():
    """ABM propagates the harmonic oscillator close to the analytic solution."""

    def f(t, y):  # noqa: ARG001
        return np.array([y[1], -y[0]])

    y0 = np.array([1.0, 0.0])
    h = 0.005
    t_final = 1.0
    t, y = _propagate_abm(f, y0, h, t_final)

    exact = np.array([np.cos(t_final), -np.sin(t_final)])
    assert np.linalg.norm(y - exact) < 1e-5


def test_abm_convergence_is_fourth_order():
    """Halving the step size shrinks the error by ~2^4 = 16 (4th-order)."""

    def f(t, y):  # noqa: ARG001
        return np.array([y[1], -y[0]])

    y0 = np.array([1.0, 0.0])
    target_t = 1.0
    errors = []
    for h in (0.02, 0.01):
        _, y = _propagate_abm(f, y0, h, target_t)
        exact = np.array([np.cos(target_t), -np.sin(target_t)])
        errors.append(np.linalg.norm(y - exact))

    ratio = errors[0] / errors[1]
    assert 10.0 < ratio < 30.0, f"ratio {ratio} not ~16 (4th order)"


def test_abm_leo_j2_matches_dop853():
    """ABM on a small fixed step matches scipy DOP853 over ~1 day (< 1e-6).

    ABM is fixed-step, so it lands on the nearest h-multiple of the target;
    we compare against DOP853's dense output at the exact t ABM reached.
    """
    rhs, y0, t_span = normalized_leo_j2(days=1.0)
    h = 0.002
    t_abm, y_abm = _propagate_abm(rhs, y0, h, t_span[1])

    sol = solve_ivp(rhs, t_span, y0, method="DOP853", rtol=1e-12, atol=1e-12, dense_output=True)
    assert sol.success
    y_ref = np.asarray(sol.sol(t_abm))
    assert np.linalg.norm(y_abm - y_ref) < 1e-6


def test_abm_step_size_change_requires_restart():
    """Changing h without re-initialising history diverges.

    With a stale (wrong-spacing) history the predictor feeds garbage derivative
    samples, so the result drifts far from the re-initialised reference. This
    documents the fixed-step contract rather than asserting a specific value.
    """
    rhs, y0, t_span = normalized_leo_j2(days=0.2)
    h = 0.01

    # Correct usage: re-initialise when the step size changes mid-propagation.
    t, y, history = initialize_abm_history(0.0, y0, h, rhs, n_stages=3)
    for _ in range(5):
        r = multistep_step(MultistepMethod.ABM, t, y, h, 1e-12, rhs, history)
        y, t, history = np.asarray(r.y_new), t + h, r.history
    # Now halve the step: must rebuild history at the new spacing.
    h2 = h / 2
    _, y_restarted, hist2 = initialize_abm_history(t, y, h2, rhs, n_stages=3)

    # Wrong usage: keep the old (h-spaced) history at the new h2 step.
    r_stale = multistep_step(MultistepMethod.ABM, t, y, h2, 1e-12, rhs, history)

    # One re-initialised step vs one stale-history step should differ markedly.
    r_good = multistep_step(MultistepMethod.ABM, t, y_restarted, h2, 1e-12, rhs, hist2)
    assert np.linalg.norm(np.asarray(r_stale.y_new) - np.asarray(r_good.y_new)) > 1e-8
