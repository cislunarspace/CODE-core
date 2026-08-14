"""CR3BP 伪势 Hessian、解析 Jacobian 与 STM 变分方程测试。"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.dynamics import pseudo_potential_hessian
from e2m2e.data.constants import Datum

pytestmark = pytest.mark.theory


@pytest.fixture
def dynamics(earth_moon_dynamics):
    return earth_moon_dynamics


@pytest.fixture
def state():
    return np.array([0.8, 0.2, 0.1, 0.0, 0.1, 0.0])


@pytest.mark.parametrize(
    "position",
    [
        (0.5, 0.0, 0.0),
        (0.8, 0.2, 0.0),
        (0.5, 0.3, 0.1),
        (1.0, 0.5, -0.3),
    ],
)
def test_pseudo_potential_hessian_is_symmetric_and_matches_finite_difference(position):
    mu = Datum.DE421.mu
    x, y, z = position
    hessian = pseudo_potential_hessian(mu, x, y, z)
    epsilon = 1e-5

    def potential(x_, y_, z_):
        r1 = np.sqrt((x_ + mu) ** 2 + y_**2 + z_**2)
        r2 = np.sqrt((x_ - 1.0 + mu) ** 2 + y_**2 + z_**2)
        return 0.5 * (x_**2 + y_**2) + (1.0 - mu) / r1 + mu / r2

    assert_allclose(hessian, hessian.T, atol=1e-14)
    dxx = (
        potential(x + epsilon, y, z) - 2.0 * potential(x, y, z) + potential(x - epsilon, y, z)
    ) / epsilon**2
    dxy = (
        potential(x + epsilon, y + epsilon, z)
        - potential(x + epsilon, y - epsilon, z)
        - potential(x - epsilon, y + epsilon, z)
        + potential(x - epsilon, y - epsilon, z)
    ) / (4.0 * epsilon**2)
    assert_allclose(hessian[0, 0], dxx, rtol=1e-4, atol=1e-8)
    assert_allclose(hessian[0, 1], dxy, rtol=1e-5)


def test_analytic_jacobian_has_the_variational_equation_structure(dynamics, state):
    jacobian = dynamics.compute_jacobian_A(state)
    hessian = pseudo_potential_hessian(dynamics.system.mu, *state[:3])

    assert_allclose(jacobian[:3, :3], np.zeros((3, 3)))
    assert_allclose(jacobian[:3, 3:], np.eye(3))
    assert_allclose(jacobian[3:, :3], hessian)
    assert jacobian[3, 4] == 2.0
    assert jacobian[4, 3] == -2.0
    assert np.trace(jacobian) == pytest.approx(0.0)


def test_analytic_jacobian_matches_the_equations_finite_difference(dynamics, state):
    jacobian = dynamics.compute_jacobian_A(state)
    epsilon = 1e-7

    for column in range(6):
        plus = state.copy()
        minus = state.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        derivative = (
            dynamics.equations_of_motion(0.0, plus) - dynamics.equations_of_motion(0.0, minus)
        ) / (2.0 * epsilon)
        assert_allclose(jacobian[:, column], derivative, rtol=1e-6, atol=1e-8)


def test_augmented_equation_uses_a_times_phi(dynamics, state):
    phi = np.eye(6)
    augmented = np.concatenate([state, phi.ravel()])
    derivative = dynamics.equations_with_stm(0.0, augmented)

    assert derivative.shape == (42,)
    assert_allclose(derivative[:6], dynamics.equations_of_motion(0.0, state))
    assert_allclose(derivative[6:].reshape(6, 6), dynamics.compute_jacobian_A(state))


def test_stm_matches_a_small_initial_state_perturbation(dynamics, state):
    t_span = (0.0, 0.5)
    reference = dynamics.propagate(state, t_span, with_stm=True)
    delta = 1e-6
    direction = np.array([1.0, -2.0, 1.0, 0.5, -1.0, 2.0])
    direction /= np.linalg.norm(direction)

    perturbed = dynamics.propagate(state + delta * direction, t_span)
    finite_difference = perturbed["states"][-1] - reference["states"][-1]
    linearized = reference["stm"][-1] @ (delta * direction)

    assert_allclose(linearized, finite_difference, rtol=1e-4, atol=1e-9)


def test_stm_preserves_phase_space_volume(dynamics, state):
    result = dynamics.propagate(state, (0.0, 1.0), with_stm=True)
    determinants = [np.linalg.det(phi) for phi in result["stm"]]
    assert_allclose(determinants, np.ones(len(determinants)), rtol=1e-4)
