from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm


@dataclass(frozen=True)
class TwoLevelMultipleShootingResult:
    t_patch: np.ndarray
    state_patch: np.ndarray
    converged: bool
    status: str
    outer_iterations: int
    level1_iterations: list[int]
    final_position_residual: float
    final_velocity_residual: float
    per_patch_position_residual: np.ndarray
    per_patch_velocity_residual: np.ndarray
    residual_history: list[tuple[float, float]]


def _build_level1_constraint(
    propagated_position: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    return propagated_position - target_position


def _build_level1_jacobian(stm: np.ndarray) -> np.ndarray:
    return stm[0:3, 3:6].copy()


def _build_level2_constraint(
    departure_velocity: np.ndarray,
    arrival_velocity: np.ndarray,
) -> np.ndarray:
    return departure_velocity - arrival_velocity


def _build_level2_patch_jacobian(
    left_stm: np.ndarray,
    right_stm: np.ndarray,
    left_departure_velocity: np.ndarray,
    left_arrival_velocity: np.ndarray,
    right_departure_velocity: np.ndarray,
    right_arrival_velocity: np.ndarray,
    left_arrival_acceleration: np.ndarray,
    right_departure_acceleration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_inverse_stm = np.linalg.pinv(left_stm)
    left_b_inverse = np.linalg.pinv(left_inverse_stm[0:3, 3:6])
    left_a = left_inverse_stm[0:3, 0:3]
    right_b_inverse = np.linalg.pinv(right_stm[0:3, 3:6])
    right_a = right_stm[0:3, 0:3]

    previous_position_block = -left_b_inverse
    previous_time_block = left_b_inverse @ left_departure_velocity
    current_position_block = left_b_inverse @ left_a - right_b_inverse @ right_a
    current_time_block = -(
        left_arrival_acceleration + (left_b_inverse @ left_a) @ left_arrival_velocity
    ) + (
        right_departure_acceleration + (right_b_inverse @ right_a) @ right_departure_velocity
    )
    next_position_block = right_b_inverse
    next_time_block = -right_b_inverse @ right_arrival_velocity

    return (
        previous_position_block,
        previous_time_block,
        current_position_block,
        current_time_block,
        next_position_block,
        next_time_block,
    )


class TwoLevelMultipleShooting:
    def __init__(self, dynamics) -> None:
        if dynamics is None:
            raise TypeError("dynamics must not be None")
        if not callable(getattr(dynamics, "propagate", None)):
            raise TypeError("dynamics must provide propagate")
        if not callable(getattr(dynamics, "equations_of_motion", None)):
            raise TypeError("dynamics must provide equations_of_motion")
        self.dynamics = dynamics

    def correct(
        self,
        t_patch: np.ndarray,
        state_patch: np.ndarray,
        *,
        max_outer_iterations: int = 10,
        max_level1_iterations: int = 20,
        position_tolerance: float = 1e-3,
        velocity_tolerance: float = 1e-6,
        boundary: str = "fixed_endpoints",
        verbose: bool = False,
    ) -> TwoLevelMultipleShootingResult:
        t_values = np.asarray(t_patch, dtype=float)
        states = np.asarray(state_patch, dtype=float)
        self._validate_inputs(
            t_values,
            states,
            max_outer_iterations,
            max_level1_iterations,
            position_tolerance,
            velocity_tolerance,
            boundary,
        )

        t_work = t_values.copy()
        state_work = states.copy()
        residual_history: list[tuple[float, float]] = []
        level1_iterations: list[int] = []
        had_level1_failure = False
        final_position = np.full(len(t_work) - 1, np.inf)
        final_velocity = np.full(len(t_work) - 1, np.inf)

        iterator = tqdm(
            range(max_outer_iterations),
            desc="Two-Level Multiple Shooting",
            unit="iter",
            disable=not verbose,
        )
        for outer_index in iterator:
            state_work, segment_iterations, had_failure = self._run_level1(
                t_work,
                state_work,
                max_level1_iterations,
                position_tolerance,
            )
            level1_iterations.extend(segment_iterations)
            had_level1_failure = had_level1_failure or had_failure
            final_position, final_velocity = self._compute_residuals(t_work, state_work)
            position_residual = float(np.sum(final_position))
            velocity_residual = float(np.sum(final_velocity))
            residual_history.append((position_residual, velocity_residual))
            if self._is_converged(
                position_residual,
                velocity_residual,
                position_tolerance,
                velocity_tolerance,
            ):
                return self._result(
                    t_work,
                    state_work,
                    True,
                    "converged",
                    outer_index + 1,
                    level1_iterations,
                    final_position,
                    final_velocity,
                    residual_history,
                )

            t_candidate, state_candidate = self._run_level2(
                t_work,
                state_work,
                position_residual + velocity_residual,
            )
            t_work = t_candidate
            state_work = state_candidate
            final_position, final_velocity = self._compute_residuals(t_work, state_work)
            residual_history[-1] = (
                float(np.sum(final_position)),
                float(np.sum(final_velocity)),
            )

        status = "level1_failed" if had_level1_failure else "max_iterations"
        return self._result(
            t_work,
            state_work,
            False,
            status,
            max_outer_iterations,
            level1_iterations,
            final_position,
            final_velocity,
            residual_history,
        )

    @staticmethod
    def _validate_inputs(
        t_values: np.ndarray,
        states: np.ndarray,
        max_outer_iterations: int,
        max_level1_iterations: int,
        position_tolerance: float,
        velocity_tolerance: float,
        boundary: str,
    ) -> None:
        if t_values.ndim != 1:
            raise ValueError("t_patch must be one-dimensional")
        if states.ndim != 2 or states.shape[1] != 6:
            raise ValueError("state_patch must have shape (n_points, 6)")
        if len(t_values) != len(states):
            raise ValueError("t_patch and state_patch must have the same length")
        if len(t_values) < 3:
            raise ValueError("at least 3 patch points are required")
        if np.any(np.diff(t_values) <= 0):
            raise ValueError("t_patch must be strictly increasing")
        if max_outer_iterations <= 0:
            raise ValueError("max_outer_iterations must be positive")
        if max_level1_iterations <= 0:
            raise ValueError("max_level1_iterations must be positive")
        if position_tolerance <= 0:
            raise ValueError("position_tolerance must be positive")
        if velocity_tolerance <= 0:
            raise ValueError("velocity_tolerance must be positive")
        if boundary != "fixed_endpoints":
            raise ValueError(f"unsupported boundary: {boundary}")

    def _run_level1(
        self,
        t_patch: np.ndarray,
        state_patch: np.ndarray,
        max_iterations: int,
        tolerance: float,
    ) -> tuple[np.ndarray, list[int], bool]:
        states = state_patch.copy()
        iterations: list[int] = []
        had_failure = False
        for segment_index in range(len(t_patch) - 1):
            velocity = states[segment_index, 3:6].copy()
            residual_norm = np.inf
            iteration_count = max_iterations
            for iteration_number in range(1, max_iterations + 1):
                initial_state = states[segment_index].copy()
                initial_state[3:6] = velocity
                result = self.dynamics.propagate(
                    initial_state,
                    (t_patch[segment_index], t_patch[segment_index + 1]),
                    with_stm=True,
                )
                final_state = np.asarray(result["states"])[-1]
                final_stm = np.asarray(result["stm"])[-1]
                residual = _build_level1_constraint(
                    final_state[:3],
                    states[segment_index + 1, :3],
                )
                residual_norm = float(np.linalg.norm(residual))
                iteration_count = iteration_number
                if residual_norm <= tolerance:
                    break
                jacobian = _build_level1_jacobian(final_stm)
                delta, _, _, _ = np.linalg.lstsq(jacobian, -residual, rcond=None)
                velocity = velocity + delta
            states[segment_index, 3:6] = velocity
            iterations.append(iteration_count)
            if residual_norm > tolerance:
                had_failure = True
        return states, iterations, had_failure

    def _run_level2(
        self,
        t_patch: np.ndarray,
        state_patch: np.ndarray,
        current_residual: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        t_next = t_patch.copy()
        states = state_patch.copy()
        n_points = len(t_next)
        n_internal = n_points - 2
        if n_internal <= 0:
            return t_next, states

        jacobian = np.zeros((3 * n_internal, 4 * n_internal))
        residuals = np.zeros(3 * n_internal)
        for patch_index in range(1, n_points - 1):
            left_result = self.dynamics.propagate(
                states[patch_index - 1],
                (t_next[patch_index - 1], t_next[patch_index]),
                with_stm=True,
            )
            right_result = self.dynamics.propagate(
                states[patch_index],
                (t_next[patch_index], t_next[patch_index + 1]),
                with_stm=True,
            )
            left_final_state = np.asarray(left_result["states"])[-1]
            right_final_state = np.asarray(right_result["states"])[-1]
            left_stm = np.asarray(left_result["stm"])[-1]
            right_stm = np.asarray(right_result["stm"])[-1]
            left_acceleration = self.dynamics.equations_of_motion(
                t_next[patch_index],
                left_final_state,
            )[3:6]
            right_acceleration = self.dynamics.equations_of_motion(
                t_next[patch_index],
                states[patch_index],
            )[3:6]
            blocks = _build_level2_patch_jacobian(
                left_stm,
                right_stm,
                states[patch_index - 1, 3:6],
                left_final_state[3:6],
                states[patch_index, 3:6],
                right_final_state[3:6],
                left_acceleration,
                right_acceleration,
            )
            row = 3 * (patch_index - 1)
            residuals[row : row + 3] = _build_level2_constraint(
                states[patch_index, 3:6],
                left_final_state[3:6],
            )
            for block_offset, position_block, time_block in (
                (-1, blocks[0], blocks[1]),
                (0, blocks[2], blocks[3]),
                (1, blocks[4], blocks[5]),
            ):
                target_patch = patch_index + block_offset
                if 0 < target_patch < n_points - 1:
                    column = 4 * (target_patch - 1)
                    jacobian[row : row + 3, column : column + 3] = position_block
                    jacobian[row : row + 3, column + 3] = time_block

        delta, _, _, _ = np.linalg.lstsq(jacobian, -residuals, rcond=None)
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate_t = t_next.copy()
            candidate_states = states.copy()
            for internal_index in range(1, n_points - 1):
                column = 4 * (internal_index - 1)
                candidate_states[internal_index, :3] = (
                    candidate_states[internal_index, :3]
                    + damping * delta[column : column + 3]
                )
                candidate_t[internal_index] = (
                    candidate_t[internal_index] + damping * delta[column + 3]
                )
            if np.any(np.diff(candidate_t) <= 0):
                continue
            position_residuals, velocity_residuals = self._compute_residuals(
                candidate_t,
                candidate_states,
            )
            candidate_residual = float(np.sum(position_residuals) + np.sum(velocity_residuals))
            if candidate_residual <= current_residual:
                return candidate_t, candidate_states
        return t_next, states

    def _compute_residuals(
        self,
        t_patch: np.ndarray,
        state_patch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        position_residuals = []
        velocity_residuals = []
        for segment_index in range(len(t_patch) - 1):
            result = self.dynamics.propagate(
                state_patch[segment_index],
                (t_patch[segment_index], t_patch[segment_index + 1]),
                with_stm=True,
            )
            final_state = np.asarray(result["states"])[-1]
            difference = final_state - state_patch[segment_index + 1]
            position_residuals.append(float(np.linalg.norm(difference[:3])))
            velocity_residuals.append(float(np.linalg.norm(difference[3:6])))
        return np.array(position_residuals), np.array(velocity_residuals)

    @staticmethod
    def _is_converged(
        position_residual: float,
        velocity_residual: float,
        position_tolerance: float,
        velocity_tolerance: float,
    ) -> bool:
        return position_residual <= position_tolerance and velocity_residual <= velocity_tolerance

    @staticmethod
    def _result(
        t_patch: np.ndarray,
        state_patch: np.ndarray,
        converged: bool,
        status: str,
        outer_iterations: int,
        level1_iterations: list[int],
        per_patch_position_residual: np.ndarray,
        per_patch_velocity_residual: np.ndarray,
        residual_history: list[tuple[float, float]],
    ) -> TwoLevelMultipleShootingResult:
        return TwoLevelMultipleShootingResult(
            t_patch=t_patch.copy(),
            state_patch=state_patch.copy(),
            converged=converged,
            status=status,
            outer_iterations=outer_iterations,
            level1_iterations=list(level1_iterations),
            final_position_residual=float(np.sum(per_patch_position_residual)),
            final_velocity_residual=float(np.sum(per_patch_velocity_residual)),
            per_patch_position_residual=per_patch_position_residual.copy(),
            per_patch_velocity_residual=per_patch_velocity_residual.copy(),
            residual_history=list(residual_history),
        )
