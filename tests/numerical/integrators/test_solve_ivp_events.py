"""Rust solve_ivp_events 的事件检测与步内求精契约。"""

import numpy as np
import pytest

from e2m2e.integrators import RkMethod, solve_ivp_events

pytestmark = pytest.mark.integrator


def test_solve_ivp_events_refines_terminal_event_in_rust():
    """Rust 事件路径应返回终止事件、事件态和步内求精时刻。"""

    def rhs(_t, state):
        return np.array([state[1], -1.0])

    def hit_ground(_t, state):
        return state[0]

    result = solve_ivp_events(
        (0.0, 5.0),
        [1.0, 0.0],
        np.linspace(0.0, 5.0, 6),
        1e-12,
        1e-12,
        rhs,
        [(hit_ground, True, 0.0)],
        method=RkMethod.PD45,
        max_step=0.01,
    )

    assert result["terminal_event"] == 0
    assert result["time"][-1] == pytest.approx(np.sqrt(2.0), abs=1e-3)
    assert result["states"][-1][0] == pytest.approx(0.0, abs=1e-3)
    assert result["t_events"][0][0] == pytest.approx(np.sqrt(2.0), abs=1e-3)
