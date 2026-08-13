"""Rust solve_ivp_events 绑定契约测试。

覆盖终止事件步内求精、非终止事件、方向过滤、多事件与空事件列表。
事件语义与 scipy solve_ivp 对齐（ADR 0023 允许的差异除外）；
经 Dynamics 编排层的事件测试见 tests/numerical/dynamics/test_dynamics_events.py。
"""

import numpy as np
import pytest

from e2m2e.integrators import RkMethod, solve_ivp_events

pytestmark = pytest.mark.integrator


def _falling(t, state):  # noqa: ARG001
    """y'' = -1：y(t) = 1 - t²/2，v(t) = -t。"""
    return np.array([state[1], -1.0])


def _falling_ground(t, state):  # noqa: ARG001
    """地面事件：y = 0 → t = √2。"""
    return state[0]


def test_terminal_event_refines_in_step():
    """终止事件应返回事件索引、步内求精时刻与事件态。"""
    result = solve_ivp_events(
        (0.0, 5.0),
        [1.0, 0.0],
        np.linspace(0.0, 5.0, 6),
        1e-12,
        1e-12,
        _falling,
        [(_falling_ground, True, 0.0)],
        method=RkMethod.PD45,
        max_step=0.01,
    )

    assert result["terminal_event"] == 0
    assert result["time"][-1] == pytest.approx(np.sqrt(2.0), abs=1e-3)
    assert result["states"][-1][0] == pytest.approx(0.0, abs=1e-3)
    assert result["t_events"][0][0] == pytest.approx(np.sqrt(2.0), abs=1e-3)


def test_nonterminal_event_continues_to_tf():
    """非终止事件记录触发时刻，积分继续到 t_span 终点。"""
    result = solve_ivp_events(
        (0.0, 5.0),
        [1.0, 0.0],
        np.linspace(0.0, 5.0, 6),
        1e-12,
        1e-12,
        _falling,
        [(_falling_ground, False, 0.0)],
        method=RkMethod.PD45,
        max_step=0.01,
    )

    assert result["terminal_event"] is None
    assert result["time"][-1] == pytest.approx(5.0)
    assert result["t_events"][0][0] == pytest.approx(np.sqrt(2.0), abs=1e-3)


def test_direction_filter_records_only_upward_crossings():
    """direction=+1 只记上行穿越（g 由负到正）。

    谐振子 y = cos t 在 [0, 2π] 内两次过零：π/2 下行（正→负）、
    3π/2 上行（负→正）。direction=+1 应只记 3π/2。
    """

    def harmonic(t, state):  # noqa: ARG001
        return np.array([state[1], -state[0]])

    def crossing(t, state):  # noqa: ARG001
        return state[0]

    result = solve_ivp_events(
        (0.0, 2.0 * np.pi),
        [1.0, 0.0],
        np.linspace(0.0, 2.0 * np.pi, 21),
        1e-12,
        1e-12,
        harmonic,
        [(crossing, False, 1.0)],
        method=RkMethod.PD45,
        max_step=0.01,
    )

    assert result["terminal_event"] is None
    assert len(result["t_events"][0]) == 1
    assert result["t_events"][0][0] == pytest.approx(3.0 * np.pi / 2.0, abs=1e-3)


def test_multiple_events_recorded_per_event():
    """多个事件各自记录在独立列表（t_events[i] 对应 events[i]）。"""

    def half_way(t, state):  # noqa: ARG001
        """y = 0.5 → t = 1。"""
        return state[0] - 0.5

    result = solve_ivp_events(
        (0.0, 5.0),
        [1.0, 0.0],
        np.linspace(0.0, 5.0, 6),
        1e-12,
        1e-12,
        _falling,
        [(_falling_ground, False, 0.0), (half_way, False, 0.0)],
        method=RkMethod.PD45,
        max_step=0.01,
    )

    assert result["terminal_event"] is None
    assert len(result["t_events"]) == 2
    assert result["t_events"][0][0] == pytest.approx(np.sqrt(2.0), abs=1e-3)
    assert result["t_events"][1][0] == pytest.approx(1.0, abs=1e-3)


def test_no_events_behaves_like_plain_integration():
    """events=[] 时等价普通积分，事件键为空。"""
    result = solve_ivp_events(
        (0.0, 5.0),
        [1.0, 0.0],
        np.linspace(0.0, 5.0, 6),
        1e-12,
        1e-12,
        _falling,
        [],
        method=RkMethod.PD45,
    )

    assert result["terminal_event"] is None
    assert len(result["t_events"]) == 0
    assert len(result["y_events"]) == 0
    assert result["time"][-1] == pytest.approx(5.0)
