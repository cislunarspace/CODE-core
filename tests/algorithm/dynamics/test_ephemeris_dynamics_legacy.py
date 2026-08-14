"""EphemerisDynamics 遗留消费者所需的最小接口契约。"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.dynamics import Dynamics, EphemerisDynamics
from e2m2e.data.constants import Datum

pytestmark = [pytest.mark.interface, pytest.mark.spice]


@pytest.fixture
def leo_state():
    radius_km = Datum.WGS84.earth_radius_km + 400.0
    circular_speed_km_s = np.sqrt(Datum.DE440.earth_gm / radius_km)
    return np.array([radius_km, 0.0, 0.0, 0.0, circular_speed_km_s, 0.0])


@pytest.fixture
def reference_et(spice_manager, reference_epoch):
    return spice_manager.utc_to_et(reference_epoch)


def test_legacy_dynamics_keeps_the_dynamics_interface(spice_eph_dynamics, spice_eph_system):
    assert isinstance(spice_eph_dynamics, Dynamics)
    assert isinstance(spice_eph_dynamics, EphemerisDynamics)
    assert spice_eph_dynamics.system is spice_eph_system


def test_legacy_equations_preserve_state_vector_layout(spice_eph_dynamics, reference_et, leo_state):
    derivative = spice_eph_dynamics.equations_of_motion(reference_et, leo_state)
    assert derivative.shape == (6,)
    assert_allclose(derivative[:3], leo_state[3:])


def test_legacy_stm_equation_starts_from_a_times_identity(
    spice_eph_dynamics, reference_et, leo_state
):
    augmented = np.concatenate([leo_state, np.eye(6).ravel()])
    derivative = spice_eph_dynamics.equations_with_stm(reference_et, augmented)
    assert derivative.shape == (42,)
    assert_allclose(
        derivative[6:].reshape(6, 6),
        spice_eph_dynamics.compute_jacobian_A(reference_et, leo_state),
    )


def test_legacy_propagation_returns_aligned_state_and_stm_histories(
    spice_eph_dynamics, reference_et, leo_state
):
    t_span = (reference_et, reference_et + 3_600.0)
    t_eval = np.linspace(*t_span, 10)
    result = spice_eph_dynamics.propagate(leo_state, t_span, t_eval=t_eval, with_stm=True)
    assert result["time"].shape == (10,)
    assert result["states"].shape == (10, 6)
    assert result["stm"].shape == (10, 6, 6)
    assert_allclose(result["states"][0], leo_state, atol=1e-9)
    assert_allclose(result["stm"][0], np.eye(6), atol=1e-9)


def test_legacy_propagation_supports_zero_and_backward_duration(
    spice_eph_dynamics, reference_et, leo_state
):
    zero = spice_eph_dynamics.propagate(leo_state, (reference_et, reference_et))
    backward = spice_eph_dynamics.propagate(leo_state, (reference_et, reference_et - 3_600.0))
    assert_allclose(zero["states"][0], leo_state, atol=1e-9)
    assert backward["time"][0] > backward["time"][-1]


def test_legacy_max_step_is_capped_by_short_propagation_duration(spice_eph_dynamics):
    assert spice_eph_dynamics._get_max_step((0.0, 1_000.0)) == pytest.approx(100.0)
    assert spice_eph_dynamics._get_max_step((0.0, 100.0)) == pytest.approx(10.0)
    assert spice_eph_dynamics._get_max_step((100.0, 0.0)) == pytest.approx(10.0)


def test_legacy_dynamics_rejects_events_explicitly(spice_eph_dynamics, reference_et, leo_state):
    def event(time, state):  # noqa: ARG001
        return float(state[0])

    with pytest.raises(ValueError, match="必须显式指定 backend"):
        spice_eph_dynamics.propagate(leo_state, (reference_et, reference_et + 100.0), events=event)
    with pytest.raises(NotImplementedError, match="事件检测"):
        spice_eph_dynamics.propagate(
            leo_state,
            (reference_et, reference_et + 100.0),
            events=event,
            backend="scipy",
        )
