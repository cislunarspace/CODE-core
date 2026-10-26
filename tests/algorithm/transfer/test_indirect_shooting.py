"""间接低推力单打靶与末端坐标行为测试。"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.transfer.indirect_shooting import (
    CartesianTarget,
    IndirectShootingConfig,
    TransformedTarget,
    solve_indirect_shooting,
    terminal_residual,
)
from e2m2e.status import ConvergenceState

pytestmark = [pytest.mark.orchestration, pytest.mark.low_thrust]


def test_cr3bp_indirect_variational_jacobian_matches_perturbed_propagation():
    """CR3BP 变分雅可比与独立扰动轨迹的线性预测一致。"""
    from e2m2e import integrators

    if integrators.propagate_cr3bp_indirect_py is None:
        pytest.skip("需要重新构建 Rust 扩展")
    state = np.array([0.8, 0.1, 0.02, 0.01, 0.2, -0.01])
    costate = np.array([0.05, -0.04, 0.03, 0.15, 0.1, -0.08])
    kwargs = {
        "t0": 0.0,
        "tf": 0.3,
        "state0": state,
        "lambda0": costate,
        "acceleration_max": 0.05,
        "epsilon": 0.2,
        "rtol": 1e-11,
        "atol": 1e-12,
        "max_step": 0.1,
        "with_jacobian": True,
    }
    result = integrators.propagate_cr3bp_indirect_py(0.0121505856, **kwargs)
    perturbation = 1e-6
    shifted = costate.copy()
    shifted[3] += perturbation
    changed = integrators.propagate_cr3bp_indirect_py(
        0.0121505856, **{**kwargs, "lambda0": shifted}
    )
    predicted = result["state"] + perturbation * np.asarray(result["jacobian"])[:, 3]
    np.testing.assert_allclose(predicted, changed["state"], rtol=1e-7, atol=1e-10)


def test_cartesian_solve_reaches_manufactured_free_transfer():
    """无自然加速度时，从已知协态生成目标并恢复同一物理交会。"""
    from e2m2e import integrators

    if integrators.propagate_indirect_py is None:
        pytest.skip("需要重新构建 Rust 扩展")
    state0 = np.zeros(6)
    known_costate = np.array([0.05, -0.04, 0.03, 1.2, 0.0, 0.0])

    def natural(_time, _state):
        return np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))

    propagated = integrators.propagate_indirect_py(
        0.0,
        1.0,
        state0,
        known_costate,
        0.1,
        0.1,
        natural,
        with_jacobian=False,
    )
    target = np.asarray(propagated["state"])
    config = IndirectShootingConfig(
        acceleration_max=0.1,
        epsilon=0.1,
        max_iterations=100,
        residual_tolerance=1e-10,
        max_step=0.2,
    )
    solution = solve_indirect_shooting(
        0.0,
        1.0,
        state0,
        np.array([0.1, 0.1, 0.1, 0.5, 0.0, 0.0]),
        CartesianTarget(target),
        config,
        natural,
    )
    assert solution.status is ConvergenceState.CONVERGED
    np.testing.assert_allclose(solution.final_state, target, atol=2e-9)


def test_transformed_target_wraps_periodic_angle_on_shortest_branch():
    """同一物理目标在角度跨 2π 分支时取最短角差。"""
    target_state = np.array([0.1, -0.2, 0.05, 0.02, 0.01, -0.03])

    def state_map(state):
        q = state[:3]
        p = state[3:]
        return np.array(
            [
                q[0] * p[0],
                np.log(abs(q[0] / p[0])),
                q[1] ** 2 + p[1] ** 2,
                np.arctan2(q[1], p[1]) % (2.0 * np.pi),
                q[2] ** 2 + p[2] ** 2,
                np.arctan2(q[2], p[2]) % (2.0 * np.pi),
            ]
        )

    target = TransformedTarget(state_map(target_state), state_map, (3, 5))
    np.testing.assert_allclose(terminal_residual(target_state, target), 0.0, atol=1e-14)
    wrapped_target = np.asarray(target.target, dtype=float).copy()
    angle = wrapped_target[3]
    wrapped_target[3] = (angle - np.pi) % (2.0 * np.pi)
    wrapped = TransformedTarget(wrapped_target, state_map, (3, 5))
    assert abs(terminal_residual(target_state, wrapped)[3]) < np.pi + 1e-14
    logarithmic_target = np.asarray(target.target, dtype=float).copy()
    logarithmic_target[1] += 2.0 * np.pi
    nonperiodic = TransformedTarget(logarithmic_target, state_map, (3, 5))
    assert terminal_residual(target_state, nonperiodic)[1] == pytest.approx(-2.0 * np.pi)


def test_failed_coordinate_transform_is_not_reported_as_convergence():
    """目标映射超出定义域时返回失败，不伪造成功末态。"""
    from e2m2e import integrators

    if integrators.propagate_cr3bp_indirect_py is None:
        pytest.skip("需要重新构建 Rust 扩展")
    target = TransformedTarget(
        np.zeros(6),
        lambda _state: (_ for _ in ()).throw(ValueError("outside element domain")),
    )
    solution = solve_indirect_shooting(
        0.0,
        0.05,
        np.array([0.8, 0.0, 0.0, 0.0, 0.2, 0.0]),
        np.zeros(6),
        target,
        IndirectShootingConfig(acceleration_max=0.1),
        cr3bp_mu=0.0121505856,
    )
    assert solution.status is ConvergenceState.FAILED
    assert not solution.converged


def test_final_iteration_physical_convergence_is_not_missed():
    """最后接受的一步满足物理容差时不得误报迭代上限。"""
    from e2m2e import integrators

    if integrators.propagate_indirect_py is None:
        pytest.skip("需要重新构建 Rust 扩展")
    known_costate = np.array([0.05, -0.04, 0.03, 1.2, 0.0, 0.0])

    def natural(_time, _state):
        return np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))

    target = np.asarray(
        integrators.propagate_indirect_py(
            0.0, 1.0, np.zeros(6), known_costate, 0.1, 0.1, natural, with_jacobian=False
        )["state"]
    )
    solution = solve_indirect_shooting(
        0.0,
        1.0,
        np.zeros(6),
        known_costate + np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CartesianTarget(target),
        IndirectShootingConfig(
            acceleration_max=0.1,
            epsilon=0.1,
            max_iterations=1,
            residual_tolerance=1e-4,
            max_step=0.2,
        ),
        natural,
    )
    assert solution.status is ConvergenceState.CONVERGED
    assert np.all(np.isfinite(solution.final_costate))
    initial_guess = known_costate + np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial_state = np.asarray(
        integrators.propagate_indirect_py(
            0.0, 1.0, np.zeros(6), initial_guess, 0.1, 0.1, natural, with_jacobian=False
        )["state"]
    )
    assert np.linalg.norm(solution.final_state - target) < 1e-4
    assert np.linalg.norm(solution.final_state - target) < np.linalg.norm(initial_state - target)
