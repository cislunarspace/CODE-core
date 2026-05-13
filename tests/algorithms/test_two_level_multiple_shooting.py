import numpy as np
import pytest

from e2m2e.algorithms import TwoLevelMultipleShooting, TwoLevelMultipleShootingResult


class LinearDynamics:
    def propagate(self, state, time_span, with_stm=True):
        dt = time_span[1] - time_span[0]
        final_state = np.asarray(state, dtype=float).copy()
        final_state[:3] = final_state[:3] + final_state[3:6] * dt
        stm = np.eye(6)
        stm[0:3, 3:6] = np.eye(3) * dt
        return {
            "time": np.array([time_span[0], time_span[1]]),
            "states": np.array([state, final_state]),
            "stm": np.array([np.eye(6), stm]),
        }

    def equations_of_motion(self, _time, state):
        derivative = np.zeros(6)
        derivative[:3] = np.asarray(state, dtype=float)[3:6]
        return derivative


class StagnantDynamics:
    def propagate(self, state, time_span, with_stm=True):
        final_state = np.asarray(state, dtype=float).copy()
        stm = np.eye(6)
        stm[0:3, 3:6] = 0.0
        return {
            "time": np.array([time_span[0], time_span[1]]),
            "states": np.array([state, final_state]),
            "stm": np.array([np.eye(6), stm]),
        }

    def equations_of_motion(self, _time, state):
        return np.zeros_like(state, dtype=float)


class TimeBendingDynamics:
    def propagate(self, state, time_span, with_stm=True):
        final_state = np.asarray(state, dtype=float).copy()
        final_state[3:6] = final_state[3:6] + np.array([10.0, 0.0, 0.0])
        stm = np.eye(6)
        stm[0:3, 3:6] = np.eye(3)
        return {
            "time": np.array([time_span[0], time_span[1]]),
            "states": np.array([state, final_state]),
            "stm": np.array([np.eye(6), stm]),
        }

    def equations_of_motion(self, _time, state):
        derivative = np.zeros(6)
        derivative[3:6] = np.array([1000.0, 0.0, 0.0])
        return derivative


class MissingPropagateDynamics:
    def equations_of_motion(self, _time, state):
        return np.zeros_like(state)


class MissingEquationsDynamics:
    def propagate(self, state, time_span, with_stm=True):
        return {"states": np.array([state, state]), "stm": np.array([np.eye(6), np.eye(6)])}


def test_two_level_multiple_shooting_is_public_algorithm():
    solver = TwoLevelMultipleShooting(LinearDynamics())

    assert solver.dynamics is not None
    assert TwoLevelMultipleShootingResult is not None


def test_correct_rejects_patch_states_not_shaped_as_n_points_by_6():
    solver = TwoLevelMultipleShooting(LinearDynamics())
    t_patch = np.array([0.0, 1.0, 2.0])
    transposed_states = np.zeros((6, 3))

    with pytest.raises(ValueError, match="state_patch"):
        solver.correct(t_patch, transposed_states)


@pytest.mark.parametrize(
    ("t_patch", "state_patch", "error"),
    [
        (np.array([[0.0, 1.0, 2.0]]), np.zeros((3, 6)), "t_patch"),
        (np.array([0.0, 1.0]), np.zeros((3, 6)), "same length"),
        (np.array([0.0, 1.0]), np.zeros((2, 6)), "at least 3"),
        (np.array([0.0, 1.0, 1.0]), np.zeros((3, 6)), "strictly increasing"),
    ],
)
def test_correct_validates_patch_point_inputs(t_patch, state_patch, error):
    solver = TwoLevelMultipleShooting(LinearDynamics())

    with pytest.raises(ValueError, match=error):
        solver.correct(t_patch, state_patch)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"max_outer_iterations": 0}, "max_outer_iterations"),
        ({"max_level1_iterations": 0}, "max_level1_iterations"),
        ({"position_tolerance": 0.0}, "position_tolerance"),
        ({"velocity_tolerance": 0.0}, "velocity_tolerance"),
        ({"boundary": "periodic"}, "unsupported boundary"),
    ],
)
def test_correct_validates_solver_options(kwargs, error):
    solver = TwoLevelMultipleShooting(LinearDynamics())
    t_patch = np.array([0.0, 1.0, 2.0])
    state_patch = np.zeros((3, 6))

    with pytest.raises(ValueError, match=error):
        solver.correct(t_patch, state_patch, **kwargs)


def test_correct_reports_level1_failure_when_segments_cannot_hit_positions():
    solver = TwoLevelMultipleShooting(StagnantDynamics())
    t_patch = np.array([0.0, 1.0, 2.0])
    state_patch = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    result = solver.correct(
        t_patch,
        state_patch,
        max_outer_iterations=2,
        max_level1_iterations=1,
        position_tolerance=1e-12,
        velocity_tolerance=1e-12,
    )

    assert result.converged is False
    assert result.status == "level1_failed"
    assert result.outer_iterations == 2
    assert result.level1_iterations == [1, 1, 1, 1]
    assert len(result.residual_history) == 2
    assert result.final_position_residual > 0.0


def test_correct_converges_linear_patch_points_without_mutating_inputs():
    solver = TwoLevelMultipleShooting(LinearDynamics())
    t_patch = np.array([0.0, 1.0, 2.0])
    state_patch = np.array(
        [
            [0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.8, 0.0, 0.0],
            [2.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    original_t_patch = t_patch.copy()
    original_state_patch = state_patch.copy()

    result = solver.correct(
        t_patch,
        state_patch,
        max_outer_iterations=3,
        max_level1_iterations=5,
        position_tolerance=1e-12,
        velocity_tolerance=1e-12,
    )

    np.testing.assert_allclose(t_patch, original_t_patch)
    np.testing.assert_allclose(state_patch, original_state_patch)
    assert result.converged is True
    assert result.status == "converged"
    assert result.state_patch.shape == (3, 6)
    assert result.t_patch.shape == (3,)
    assert result.outer_iterations >= 1
    assert result.final_position_residual <= 1e-12
    assert result.final_velocity_residual <= 1e-12
    assert result.per_patch_position_residual.shape == (2,)
    assert result.per_patch_velocity_residual.shape == (2,)
    assert result.residual_history
    np.testing.assert_allclose(result.state_patch[0, :3], state_patch[0, :3])
    np.testing.assert_allclose(result.state_patch[-1, :3], state_patch[-1, :3])
    assert result.t_patch[0] == t_patch[0]
    assert result.t_patch[-1] == t_patch[-1]


def test_correct_preserves_strictly_increasing_times_after_level2_attempts():
    solver = TwoLevelMultipleShooting(TimeBendingDynamics())
    t_patch = np.array([0.0, 1.0, 2.0])
    state_patch = np.zeros((3, 6))

    result = solver.correct(
        t_patch,
        state_patch,
        max_outer_iterations=1,
        max_level1_iterations=1,
        position_tolerance=1e-12,
        velocity_tolerance=1e-12,
    )

    assert np.all(np.diff(result.t_patch) > 0)


@pytest.mark.parametrize("dynamics", [MissingPropagateDynamics(), MissingEquationsDynamics()])
def test_constructor_requires_dynamics_protocol(dynamics):
    with pytest.raises(TypeError, match="dynamics"):
        TwoLevelMultipleShooting(dynamics)
