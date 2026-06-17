"""积分器族端到端对比测试（Slice 9 闭包）。

验证 PD45/PD78/RK89 与 DOP853 一致性，
以及高阶方法步数更少。
"""

import numpy as np
from scipy.integrate import solve_ivp

from e2m2e.integrators import RkMethod
from tests.integrators.conftest import normalized_leo_j2, propagate_rk


def test_rk_family_matches_dop853():
    """PD45/PD78/RK89 each match scipy DOP853 on LEO + J2 over 1 day (< 1e-9)."""
    rhs, y0, t_span = normalized_leo_j2(days=1.0)
    sol = solve_ivp(
        rhs, t_span, y0, method="DOP853", rtol=1e-13, atol=1e-13, dense_output=True
    )
    assert sol.success

    for method in (RkMethod.PD45, RkMethod.PD78, RkMethod.RK89):
        t, y, _ = propagate_rk(method, rhs, y0, t_span, tol=1e-13, h0=0.01)
        y_ref = np.asarray(sol.sol(t))
        err = np.linalg.norm(y - y_ref)
        assert err < 1e-9, f"{method} vs DOP853 error {err} too large"


def test_rk_family_mutual_consistency():
    """PD45/PD78/RK89 agree with each other on LEO + J2 over 1 day (< 1e-9)."""
    rhs, y0, t_span = normalized_leo_j2(days=1.0)
    _, y_pd45, _ = propagate_rk(RkMethod.PD45, rhs, y0, t_span, tol=1e-13, h0=0.01)
    _, y_pd78, _ = propagate_rk(RkMethod.PD78, rhs, y0, t_span, tol=1e-13, h0=0.01)
    _, y_rk89, _ = propagate_rk(RkMethod.RK89, rhs, y0, t_span, tol=1e-13, h0=0.01)

    assert np.linalg.norm(y_pd45 - y_pd78) < 1e-9
    assert np.linalg.norm(y_pd78 - y_rk89) < 1e-9
    assert np.linalg.norm(y_pd45 - y_rk89) < 1e-9


def test_higher_order_uses_fewer_steps():
    """PD78 (order 8) and RK89 (order 9) use far fewer steps than PD45 (order 5).

    RK89 (Verner 9(8)) carries a larger error constant than DOP8(7); in this
    tolerance band it does not necessarily beat PD78 on step count, so we only
    assert the unambiguous gap between 5th order and the high-order pair.
    """
    rhs, y0, t_span = normalized_leo_j2(days=1.0)
    _, _, n_pd45 = propagate_rk(RkMethod.PD45, rhs, y0, t_span, tol=1e-13, h0=0.01)
    _, _, n_pd78 = propagate_rk(RkMethod.PD78, rhs, y0, t_span, tol=1e-13, h0=0.01)
    _, _, n_rk89 = propagate_rk(RkMethod.RK89, rhs, y0, t_span, tol=1e-13, h0=0.01)

    assert n_pd78 < n_pd45 / 5
    assert n_rk89 < n_pd45 / 5
