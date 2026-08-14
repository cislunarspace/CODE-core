"""Dynamics 公开事件接口的契约测试。"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import BCR4BP_Dynamics, BCR4BPSystem
from e2m2e.algorithm.manifold.sections import PoincareSection, detect_crossings

pytestmark = pytest.mark.interface


@pytest.fixture(params=["cr3bp", "bcr4bp"])
def dynamics(request, earth_moon_dynamics):
    if request.param == "cr3bp":
        return earth_moon_dynamics
    return BCR4BP_Dynamics(BCR4BPSystem.earth_moon())


@pytest.fixture
def off_plane_state():
    return np.array([0.8, 0.05, 0.0, 0.0, 0.0, 0.0])


def test_section_event_exposes_scipy_event_attributes():
    event = PoincareSection.plane(axis=1, value=0.0).event(direction=-1, terminal=True)
    assert callable(event)
    assert event.direction == -1
    assert event.terminal is True
    assert event(0.0, np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])) == pytest.approx(0.5)


def test_events_require_an_explicit_backend(dynamics, off_plane_state):
    event = PoincareSection.plane(axis=1, value=0.0).event(direction=-1)
    with pytest.raises(ValueError, match="backend"):
        dynamics.propagate(off_plane_state, (0.0, 1.0), events=[event])


@pytest.mark.parametrize("backend", ["auto", "gpu"])
def test_implicit_or_unknown_backends_are_rejected(dynamics, off_plane_state, backend):
    with pytest.raises(ValueError, match="backend"):
        dynamics.propagate(off_plane_state, (0.0, 1.0), backend=backend)


def test_scipy_terminal_event_stops_at_the_event_surface(dynamics, off_plane_state):
    event = PoincareSection.plane(axis=1, value=0.0).event(direction=-1, terminal=True)
    result = dynamics.propagate(off_plane_state, (0.0, 10.0), events=[event], backend="scipy")

    assert len(result["t_events"][0]) == 1
    assert result["time"][-1] == result["t_events"][0][-1]
    assert result["time"][-1] < 10.0
    assert abs(result["y_events"][0][0][1]) < 1e-10
    assert result["y_events"][0][0][4] < 0.0


def test_scipy_direction_filter_records_only_matching_crossings(dynamics, off_plane_state):
    section = PoincareSection.plane(axis=1, value=0.0)
    result = dynamics.propagate(
        off_plane_state,
        (0.0, 10.0),
        events=[section.event(direction=-1), section.event(direction=1)],
        backend="scipy",
    )
    down_times, up_times = result["t_events"]
    down_states, up_states = result["y_events"]

    assert len(down_times) > 0
    assert len(up_times) > 0
    assert down_times[0] < up_times[0]
    assert np.all(np.abs(down_states[:, 1]) < 1e-10)
    assert np.all(down_states[:, 4] < 0.0)
    assert np.all(np.abs(up_states[:, 1]) < 1e-10)
    assert np.all(up_states[:, 4] > 0.0)


def test_scipy_event_times_agree_with_post_hoc_section_detection(dynamics, off_plane_state):
    section = PoincareSection.plane(axis=1, value=0.0)
    result = dynamics.propagate(
        off_plane_state,
        (0.0, 10.0),
        t_eval=np.linspace(0.0, 10.0, 2001),
        events=[section.event(direction=-1)],
        backend="scipy",
    )
    crossings = detect_crossings(result["time"], result["states"], section)
    post_hoc_times = np.array([time for time, state, _ in crossings if state[4] < 0.0])
    np.testing.assert_allclose(result["t_events"][0], post_hoc_times, atol=1e-5)


def test_events_accept_a_single_callable_and_support_augmented_stm_state(dynamics, off_plane_state):
    event = PoincareSection.plane(axis=1, value=0.0).event(direction=-1, terminal=True)
    result = dynamics.propagate(
        off_plane_state,
        (0.0, 10.0),
        with_stm=True,
        events=event,
        backend="scipy",
    )
    assert result["y_events"][0].shape == (1, 42)
    assert result["stm"].shape[1:] == (6, 6)


def test_empty_event_list_has_no_event_result_keys(dynamics, off_plane_state):
    result = dynamics.propagate(off_plane_state, (0.0, 1.0), events=[])
    assert "t_events" not in result
    assert "y_events" not in result


def test_rust_event_backend_keeps_the_terminal_event_contract(dynamics, off_plane_state):
    event = PoincareSection.plane(axis=1, value=0.0).event(direction=-1, terminal=True)
    result = dynamics.propagate(off_plane_state, (0.0, 10.0), events=[event], backend="rust")
    assert len(result["t_events"][0]) == 1
    assert result["time"][-1] == result["t_events"][0][-1]
    assert abs(result["y_events"][0][0][1]) < 1e-6
