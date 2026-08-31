"""CR3BP/BCR4BP rust 事件路径 EOM 下沉（issue #594）的编排层等价测试。

``backend="rust"`` 事件传播的每步 RHS 求值留在 Rust 内（``RustEomKernel``
内核分派，复用 ``e2m2e-forces`` 的 EOM/STM 实现）；等价参照 = 通用
``solve_ivp_events`` + Python EOM 回调（issue #594 的对拍要求）。
事件语义本身（步内二分、无稠密输出，ADR 0020 决策 4）不在本文件断言范围，
见 ``tests/algorithm/dynamics/test_events.py``。
"""

from collections.abc import Callable

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import BCR4BP_Dynamics, BCR4BPSystem
from e2m2e.integrators import solve_ivp_events

pytestmark = pytest.mark.interface

T_SPAN = (0.0, 6.0)
OFF_PLANE_STATE = np.array([0.8, 0.05, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture(params=["cr3bp", "bcr4bp"])
def dynamics(request, earth_moon_dynamics):
    if request.param == "cr3bp":
        return earth_moon_dynamics
    return BCR4BP_Dynamics(BCR4BPSystem.earth_moon())


@pytest.fixture
def events() -> list[Callable]:
    """非终止 y=0 截面事件（保持积分全程，便于逐点对拍）。"""

    def g(t, state):
        return state[1]

    return [g]


def _reference_route(dyn, initial_state, t_eval, with_stm, events):
    """参照路线：通用 solve_ivp_events + Python EOM 回调（不经内核分派）。"""
    y0 = np.concatenate([initial_state, np.eye(6).ravel()]) if with_stm else initial_state
    specs = [
        (g, bool(getattr(g, "terminal", False)), float(getattr(g, "direction", 0))) for g in events
    ]
    return solve_ivp_events(
        T_SPAN,
        y0,
        t_eval,
        dyn.rtol,
        dyn.atol,
        dyn._get_eom_func(with_stm=with_stm),
        specs,
        max_step=dyn._get_max_step(T_SPAN),
        state_error_dim=6 if with_stm else None,
    )


@pytest.mark.parametrize("with_stm", [False, True])
def test_rust_event_path_matches_python_eom_reference(dynamics, events, with_stm):
    """内核分派结果与参照路线逐点等价：states/STM/事件时刻/事件态。"""
    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], 61)
    got = dynamics.propagate(
        OFF_PLANE_STATE, T_SPAN, t_eval=t_eval, with_stm=with_stm, events=events, backend="rust"
    )
    ref = _reference_route(dynamics, OFF_PLANE_STATE, t_eval, with_stm, events)

    n = 6
    ref_states = np.asarray(ref["states"])
    if with_stm:
        assert got["states"].shape == (len(t_eval), 6)
        assert got["stm"].shape == (len(t_eval), 6, 6)
        np.testing.assert_allclose(got["states"], ref_states[:, :n], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(
            got["stm"],
            ref_states[:, n:].reshape(-1, n, n),
            rtol=1e-8,
            atol=1e-12,
        )
    else:
        assert got["states"].shape == (len(t_eval), 6)
        np.testing.assert_allclose(got["states"], ref_states, rtol=1e-9, atol=1e-12)

    assert len(got["t_events"][0]) > 0
    np.testing.assert_allclose(got["t_events"][0], ref["t_events"][0], rtol=0, atol=1e-8)
    for got_ye, ref_ye in zip(got["y_events"], ref["y_events"], strict=True):
        np.testing.assert_allclose(
            np.asarray(got_ye)[:, :n], np.asarray(ref_ye)[:, :n], rtol=1e-8, atol=1e-12
        )


def test_rust_event_path_side_effects_and_contract(dynamics, events):
    """last_trajectory/last_stm 副作用与结果契约保持不变。"""
    got = dynamics.propagate(OFF_PLANE_STATE, T_SPAN, with_stm=True, events=events, backend="rust")
    side_t, side_x = dynamics.last_trajectory
    assert side_t.shape == got["time"].shape
    np.testing.assert_allclose(side_t, got["time"])
    np.testing.assert_allclose(side_x, got["states"])
    assert dynamics.last_stm.shape == (got["time"].shape[0], 6, 6)
    assert isinstance(got["t_events"], list)
    assert isinstance(got["y_events"], list)
    assert len(got["t_events"]) == len(events)
    assert np.asarray(got["y_events"][0]).shape[1] == 42


def test_rust_event_path_with_jacobi_cr3bp(earth_moon_dynamics, events):
    """CR3BP with_jacobi=True：jacobi 键仍沿轨迹逐点输出。"""
    got = earth_moon_dynamics.propagate(
        OFF_PLANE_STATE,
        T_SPAN,
        with_jacobi=True,
        events=events,
        backend="rust",
    )
    jacobi = np.asarray(got["jacobi"])
    assert jacobi.shape == (np.asarray(got["states"]).shape[0],)
    assert np.all(np.isfinite(jacobi))
    assert earth_moon_dynamics.jacobi_error < 1e-8


def test_bcr4bp_with_jacobi_still_rejected(events):
    """BCR4BP 无 Jacobi 积分：with_jacobi=True 维持 NotImplementedError。"""
    dyn = BCR4BP_Dynamics(BCR4BPSystem.earth_moon())
    with pytest.raises(NotImplementedError):
        dyn.propagate(OFF_PLANE_STATE, T_SPAN, with_jacobi=True, events=events, backend="rust")


def test_terminal_event_truncation_matches_reference(dynamics, events):
    """terminal 事件截断：末点 = 事件点，与参照路线的事件时刻一致。"""

    def terminal_y(t, state):
        return state[1]

    terminal_y.terminal = True
    terminal_y.direction = -1.0

    got = dynamics.propagate(
        OFF_PLANE_STATE,
        T_SPAN,
        t_eval=np.linspace(*T_SPAN, 61),
        with_stm=True,
        events=[terminal_y],
        backend="rust",
    )
    ref = _reference_route(dynamics, OFF_PLANE_STATE, np.linspace(*T_SPAN, 61), True, [terminal_y])
    # 两侧 t_eval 一致 → 步长序列一致，事件时刻应到 1e-8 内一致；
    # 输出末点被截断为求精后的事件点。
    assert got["time"][-1] == pytest.approx(got["t_events"][0][-1], abs=1e-12)
    assert got["time"][-1] < T_SPAN[1]
    np.testing.assert_allclose(got["t_events"][0], ref["t_events"][0], rtol=0, atol=1e-8)
