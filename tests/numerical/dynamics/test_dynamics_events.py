"""Dynamics.propagate 事件检测测试（scipy solve_ivp 事件语义）。

覆盖 xz 平面（y=0）穿越：terminal 截断、direction 过滤、与事后检测
（sections.detect_crossings）的一致性，以及 STM 增广传播下的事件。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.manifold.sections import PoincareSection, detect_crossings
from e2m2e.data.constants import Datum

pytestmark = pytest.mark.theory


@pytest.fixture
def dynamics():
    """Create Earth-Moon CR3BP dynamics using DE421 datum."""
    system = CR3BP_System(
        mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
    )._with_default_scales()
    return CR3BP_Dynamics(system)


@pytest.fixture
def y0_off_plane():
    """初值在 xz 平面外（y > 0、vy = 0），保证先下行穿越 y=0。"""
    return np.array([0.8, 0.05, 0.0, 0.0, 0.0, 0.0])


def test_section_event_attributes():
    """PoincareSection.event 生成带 scipy 属性的 callable。"""
    section = PoincareSection.plane(axis=1, value=0.0)
    event = section.event(direction=-1, terminal=True)

    assert callable(event)
    assert event.direction == -1
    assert event.terminal is True
    assert event(0.0, np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])) == pytest.approx(0.5)


def test_terminal_event_stops_at_xz_plane(dynamics, y0_off_plane):
    """terminal 事件在首次下行穿越 xz 平面（y=0）时截断积分。"""
    section = PoincareSection.plane(axis=1, value=0.0)
    event = section.event(direction=-1, terminal=True)

    result = dynamics.propagate(y0_off_plane, (0.0, 10.0), events=[event], backend="scipy")

    t_events = result["t_events"][0]
    y_events = result["y_events"][0]
    assert len(t_events) == 1
    # scipy terminal 语义：轨迹末点即事件点，积分被截断
    assert result["time"][-1] == t_events[-1]
    assert result["time"][-1] < 10.0
    # 事件点落在截面上，且为下行穿越（vy < 0）
    assert abs(y_events[0][1]) < 1e-10
    assert y_events[0][4] < 0


def test_direction_filter(dynamics, y0_off_plane):
    """direction 过滤：下行/上行事件各自只记对应方向的穿越。"""
    section = PoincareSection.plane(axis=1, value=0.0)
    down = section.event(direction=-1)
    up = section.event(direction=1)

    result = dynamics.propagate(y0_off_plane, (0.0, 10.0), events=[down, up], backend="scipy")

    t_down, t_up = result["t_events"]
    y_down, y_up = result["y_events"]
    assert len(t_down) > 0
    assert len(t_up) > 0
    # 从 y > 0 出发先下行穿越
    assert t_down[0] < t_up[0]
    # 穿越态在截面上，速度方向与 direction 一致
    assert np.all(np.abs(y_down[:, 1]) < 1e-10)
    assert np.all(y_down[:, 4] < 0)
    assert np.all(np.abs(y_up[:, 1]) < 1e-10)
    assert np.all(y_up[:, 4] > 0)


def test_event_matches_post_hoc_detection(dynamics, y0_off_plane):
    """积分中检测与事后 detect_crossings 的穿越时刻一致。"""
    section = PoincareSection.plane(axis=1, value=0.0)
    t_eval = np.linspace(0.0, 10.0, 2001)

    result = dynamics.propagate(
        y0_off_plane,
        (0.0, 10.0),
        t_eval=t_eval,
        events=[section.event(direction=-1)],
        backend="scipy",
    )

    post_hoc = detect_crossings(result["time"], result["states"], section)
    t_post_down = np.array([t for t, state, _ in post_hoc if state[4] < 0])
    t_event_down = np.asarray(result["t_events"][0])

    assert len(t_event_down) == len(t_post_down)
    # 事后检测基于密采样点的线性插值，自身误差 ~dt²，容差取 1e-5
    np.testing.assert_allclose(t_event_down, t_post_down, atol=1e-5)


def test_single_event_callable_accepted(dynamics, y0_off_plane):
    """events 可传单个 callable（scipy 风格），自动包装为列表。"""
    section = PoincareSection.plane(axis=1, value=0.0)

    result = dynamics.propagate(
        y0_off_plane, (0.0, 10.0), events=section.event(direction=-1), backend="scipy"
    )

    assert len(result["t_events"]) == 1
    assert len(result["t_events"][0]) > 0


def test_events_with_stm(dynamics, y0_off_plane):
    """STM 增广传播下事件函数接收 42 维状态；section.event 自动截取前 6 维。"""
    section = PoincareSection.plane(axis=1, value=0.0)
    event = section.event(direction=-1, terminal=True)

    result = dynamics.propagate(
        y0_off_plane, (0.0, 10.0), with_stm=True, events=[event], backend="scipy"
    )

    assert len(result["t_events"][0]) == 1
    # y_events 携带增广状态（6 + 36）
    assert result["y_events"][0].shape == (1, 42)
    assert result["stm"].shape[1:] == (6, 6)
    assert result["time"][-1] == result["t_events"][0][-1]


def test_no_events_no_event_keys(dynamics, y0_off_plane):
    """不传 events 时返回字典不含事件键（保持原契约）。"""
    result = dynamics.propagate(y0_off_plane, (0.0, 1.0))

    assert "t_events" not in result
    assert "y_events" not in result


# ---------------------------------------------------------------------------
# backend 参数（ADR 0020 决策 4：能力缺失显式选择，不传报错、拒绝 auto）
# ---------------------------------------------------------------------------


def test_events_without_backend_raises(dynamics, y0_off_plane):
    """events 非 None 时不传 backend 必须报错（不允许隐式选择）。"""
    section = PoincareSection.plane(axis=1, value=0.0)
    with pytest.raises(ValueError, match="backend"):
        dynamics.propagate(y0_off_plane, (0.0, 10.0), events=[section.event(direction=-1)])


def test_backend_auto_rejected(dynamics, y0_off_plane):
    """backend='auto' 一律拒绝（auto 仍是隐式选择）。"""
    section = PoincareSection.plane(axis=1, value=0.0)
    with pytest.raises(ValueError, match="backend"):
        dynamics.propagate(
            y0_off_plane, (0.0, 10.0), events=[section.event(direction=-1)], backend="auto"
        )
    with pytest.raises(ValueError, match="backend"):
        dynamics.propagate(y0_off_plane, (0.0, 10.0), backend="auto")


def test_backend_invalid_value_rejected(dynamics, y0_off_plane):
    """backend 只接受 'scipy'/'rust'。"""
    with pytest.raises(ValueError, match="backend"):
        dynamics.propagate(y0_off_plane, (0.0, 10.0), backend="gpu")


def test_empty_events_equivalent_to_none(dynamics, y0_off_plane):
    """events=[] 等价于无事件：不触发事件分支，结果不含事件键。"""
    result = dynamics.propagate(y0_off_plane, (0.0, 1.0), events=[])

    assert "t_events" not in result
    assert "y_events" not in result
    assert len(result["states"]) > 0


def test_backend_ignored_without_events(dynamics, y0_off_plane):
    """无 events 时 backend 被忽略（rust 快速路径为唯一路径）。"""
    result = dynamics.propagate(y0_off_plane, (0.0, 1.0), backend="scipy")

    assert "t_events" not in result
    assert len(result["states"]) > 0


# ---------------------------------------------------------------------------
# rust 事件积分路径（solve_ivp_events）
# 事件时刻为接受步端点线性插值 + 二分求精（无稠密输出），与 scipy 根求解
# 语义未完全对齐——这里断言 rust 自身语义，不做与 scipy 时刻的逐点对比。
# ---------------------------------------------------------------------------


def test_rust_terminal_event_stops_at_xz_plane(dynamics, y0_off_plane):
    """rust 路径：terminal 事件触发即截断，轨迹末点即事件点。"""
    section = PoincareSection.plane(axis=1, value=0.0)
    event = section.event(direction=-1, terminal=True)

    result = dynamics.propagate(y0_off_plane, (0.0, 10.0), events=[event], backend="rust")

    t_events = result["t_events"][0]
    y_events = result["y_events"][0]
    assert len(t_events) == 1
    assert result["time"][-1] == t_events[-1]
    assert result["time"][-1] < 10.0
    # 事件点为步内线性插值二分求精，落点逼近截面（非真实轨迹点，容差放宽）
    assert abs(y_events[0][1]) < 1e-6
    assert y_events[0][4] < 0


def test_rust_direction_filter(dynamics, y0_off_plane):
    """rust 路径：direction 过滤下行/上行穿越各自记录。"""
    section = PoincareSection.plane(axis=1, value=0.0)
    down = section.event(direction=-1)
    up = section.event(direction=1)

    result = dynamics.propagate(y0_off_plane, (0.0, 10.0), events=[down, up], backend="rust")

    t_down, t_up = result["t_events"]
    y_down, y_up = result["y_events"]
    assert len(t_down) > 0
    assert len(t_up) > 0
    assert t_down[0] < t_up[0]
    assert np.all(np.abs(y_down[:, 1]) < 1e-6)
    assert np.all(y_down[:, 4] < 0)
    assert np.all(np.abs(y_up[:, 1]) < 1e-6)
    assert np.all(y_up[:, 4] > 0)


def test_rust_events_with_stm(dynamics, y0_off_plane):
    """rust 路径：STM 增广传播下事件函数接收 42 维状态。"""
    section = PoincareSection.plane(axis=1, value=0.0)
    event = section.event(direction=-1, terminal=True)

    result = dynamics.propagate(
        y0_off_plane, (0.0, 10.0), with_stm=True, events=[event], backend="rust"
    )

    assert len(result["t_events"][0]) == 1
    assert result["y_events"][0].shape == (1, 42)
    assert result["stm"].shape[1:] == (6, 6)
    assert result["time"][-1] == result["t_events"][0][-1]
