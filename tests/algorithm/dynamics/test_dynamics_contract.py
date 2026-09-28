"""Dynamics 公开传播、截面和碰撞配置的接口契约测试。"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.dynamics.dynamics import Dynamics, finalize_rust_propagation
from e2m2e.data.constants import MOON, Datum
from e2m2e.exceptions import PropagationFailure

pytestmark = pytest.mark.interface


@pytest.fixture
def dynamics(earth_moon_dynamics):
    return earth_moon_dynamics


@pytest.fixture
def sample_state():
    return np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])


def test_dynamics_initializes_the_default_integrator_contract(dynamics):
    assert isinstance(dynamics, CR3BP_Dynamics)
    assert dynamics.rtol == 1e-12
    assert dynamics.atol == 1e-12


def test_propagation_returns_an_aligned_state_history(dynamics, sample_state):
    t_eval = np.linspace(0.0, 1.0, 11)
    result = dynamics.propagate(sample_state, (0.0, 1.0), t_eval=t_eval)
    assert result["time"].shape == (len(t_eval),)
    assert result["states"].shape == (len(t_eval), 6)
    assert_allclose(result["time"], t_eval)


def test_stm_propagation_returns_an_aligned_history(dynamics, sample_state):
    result = dynamics.propagate(sample_state, (0.0, 1.0), with_stm=True)
    assert result["stm"].shape == (len(result["states"]), 6, 6)
    assert_allclose(result["stm"][0], np.eye(6), atol=1e-12)


def test_cross_section_predicate_uses_the_requested_coordinate(dynamics):
    state = np.array([0.5, 0.3, 0.1, 0.0, 0.0, 0.0])
    assert dynamics.check_cross_section(state, "x", 0.5)
    assert dynamics.check_cross_section(state, "y", 0.3)
    assert dynamics.check_cross_section(state, "z", 0.1)
    with pytest.raises(ValueError, match="无效的平面"):
        dynamics.check_cross_section(state, "invalid", 0.0)


def test_collision_detection_requires_injected_body_radii(sample_state):
    dynamics = CR3BP_Dynamics(
        CR3BP_System(mu=Datum.DE421.mu, primary="Earth", secondary="Moon")._with_default_scales()
    )
    with pytest.raises(ValueError, match="body-radius"):
        dynamics.propagate(
            sample_state,
            (0.0, 1.0),
            backend="scipy",
            collision_detection=True,
        )


def test_collision_detection_requires_an_explicit_backend(dynamics, sample_state):
    dynamics.system.primary_radius_km = Datum.WGS84.earth_radius_km
    dynamics.system.secondary_radius_km = MOON.require_mean_radius_km()
    with pytest.raises(ValueError, match="backend"):
        dynamics.propagate(sample_state, (0.0, 1.0), collision_detection=True)


def test_collision_detection_is_disabled_by_default(dynamics, sample_state):
    result = dynamics.propagate(sample_state, (0.0, 1.0))
    assert "collision" not in result


def test_finalize_rust_propagation_raises_on_truncated_output_without_caching():
    owner = Dynamics.__new__(Dynamics)
    owner.last_trajectory = None
    owner.last_stm = None
    with pytest.raises(RuntimeError, match="returned 1 of 3 requested time points"):
        finalize_rust_propagation(
            owner,
            {"states": [[0.0] * 6], "time": [0.0]},
            [0.0, 1.0, 2.0],
            with_stm=False,
            error=RuntimeError,
            label="Rust propagation",
        )
    assert owner.last_trajectory is None


def test_finalize_rust_propagation_raises_with_the_injected_error_type():
    owner = Dynamics.__new__(Dynamics)
    with pytest.raises(PropagationFailure, match="Rust STM propagation returned"):
        finalize_rust_propagation(
            owner,
            {"states": [[0.0] * 6], "time": [0.0]},
            [0.0, 1.0, 2.0],
            with_stm=False,
            error=PropagationFailure,
            label="Rust STM propagation",
        )


def test_finalize_rust_propagation_assembles_outputs_and_caches_trajectory():
    owner = Dynamics.__new__(Dynamics)
    stm = np.eye(6).reshape(1, 6, 6).repeat(2, axis=0)
    out = finalize_rust_propagation(
        owner,
        {
            "states": [[0.0] * 6, [1.0] * 6],
            "time": [0.0, 1.0],
            "stm": stm.reshape(-1).tolist(),
        },
        [0.0, 1.0],
        with_stm=True,
        error=RuntimeError,
        label="Rust STM propagation",
    )
    assert out["states"].shape == (2, 6)
    assert out["stm"].shape == (2, 6, 6)
    assert owner.last_trajectory[0].tolist() == [0.0, 1.0]
    assert owner.last_stm.shape == (2, 6, 6)
