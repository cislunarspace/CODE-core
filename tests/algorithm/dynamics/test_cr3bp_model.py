"""CR3BP 运动方程、Jacobi 不变量与传播接口测试。"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.family.cr3bp_orbits import design_dpo

pytestmark = pytest.mark.theory


@pytest.fixture
def dynamics(earth_moon_dynamics):
    return earth_moon_dynamics


@pytest.fixture
def sample_state():
    return np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])


def test_state_equations_have_kinematic_position_derivatives(dynamics):
    state = np.array([0.5, -0.2, 0.1, 1.0, 2.0, 3.0])
    derivative = dynamics.equations_of_motion(0.0, state)
    assert_allclose(derivative[:3], state[3:])


def test_zero_z_state_stays_in_the_xy_plane(dynamics):
    state = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
    derivative = dynamics.equations_of_motion(0.0, state)
    assert derivative[2] == 0.0
    assert derivative[5] == 0.0


def test_pseudo_potential_acceleration_is_symmetric_under_y_reflection(dynamics):
    state_plus = np.array([0.8, 0.1, 0.0, 0.0, 0.0, 0.0])
    state_minus = np.array([0.8, -0.1, 0.0, 0.0, 0.0, 0.0])
    derivative_plus = dynamics.equations_of_motion(0.0, state_plus)
    derivative_minus = dynamics.equations_of_motion(0.0, state_minus)

    assert derivative_plus[3] == pytest.approx(derivative_minus[3])
    assert derivative_plus[4] == pytest.approx(-derivative_minus[4])
    assert derivative_plus[5] == pytest.approx(derivative_minus[5])


def test_equations_remain_finite_at_machine_scale_singularity(dynamics):
    mu = dynamics.system.mu
    for x in (-mu, 1.0 - mu):
        derivative = dynamics.equations_of_motion(0.0, np.array([x, 0.0, 0.0, 0.0, 0.1, 0.0]))
        assert np.all(np.isfinite(derivative))


def test_jacobi_constant_is_conserved_over_a_periodic_orbit_period(dynamics):
    """REQ-003：周期 DPO 在完整周期内的 Jacobi 最大漂移不超过 1e-10。

    ADR 0037 预算内：振幅取 25000 km——_walk_family 的种子点附近，design_dpo
    一次修正即命中（20000 km 需远离种子行走 ~10s，为生产侧下限）；tol_km 放宽
    至 1000 只控振幅命中精度，每个 correct_at 仍返回完全收敛的周期轨道，守恒
    语义不变。设计省下的预算把 t_eval 采样密度恢复为 1000。
    """
    orbit = design_dpo(25_000.0, tol_km=1000.0)
    result = dynamics.propagate(
        orbit.states[0],
        (0.0, orbit.period),
        t_eval=np.linspace(0.0, orbit.period, 1000),
        with_jacobi=True,
    )
    jacobi = np.asarray(result["jacobi"])
    assert np.max(np.abs(jacobi - jacobi[0])) < 1e-10


def test_jacobi_constant_implementation_agrees_with_system(dynamics, sample_state):
    assert dynamics.compute_jacobi_constant(sample_state) == pytest.approx(
        dynamics.system.get_jacobi_constant(sample_state)
    )
