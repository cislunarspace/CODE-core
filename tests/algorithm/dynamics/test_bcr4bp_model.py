"""BCR4BP 方程、变分方程及其相对 CR3BP 的退化测试。"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.dynamics import BCR4BP_Dynamics, BCR4BPSystem, CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.dynamics.potential import pseudo_potential_hessian
from e2m2e.data.constants import SECONDS_PER_DAY, Datum

pytestmark = pytest.mark.theory


@pytest.fixture
def sample_state():
    return np.array([1.1202109158830986, 0.0, 0.0, 0.0, -0.46178983697629084, 0.0])


@pytest.fixture
def bcr4bp_dynamics():
    return BCR4BP_Dynamics(BCR4BPSystem.earth_moon())


@pytest.fixture
def cr3bp_dynamics():
    system = CR3BP_System(
        mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
    )._with_default_scales()
    return CR3BP_Dynamics(system)


@pytest.fixture
def zero_sun_dynamics():
    system = BCR4BPSystem.earth_moon()
    system.sun_mass = 0.0
    return BCR4BP_Dynamics(system)


def test_zero_solar_mass_reduces_bcr4bp_to_cr3bp(zero_sun_dynamics, cr3bp_dynamics, sample_state):
    states = [
        sample_state,
        np.array([0.8, 0.1, 0.05, 0.2, 0.1, -0.03]),
        np.array([-0.5, -0.4, 0.1, 0.0, 0.3, 0.02]),
    ]
    for state in states:
        for time in (0.0, 0.7, 3.3):
            assert_allclose(
                zero_sun_dynamics.equations_of_motion(time, state),
                cr3bp_dynamics.equations_of_motion(time, state),
                atol=1e-14,
            )
            assert_allclose(
                zero_sun_dynamics.compute_jacobian_A(time, state),
                cr3bp_dynamics.compute_jacobian_A(state),
                atol=1e-14,
            )


def test_zero_solar_mass_propagation_matches_cr3bp(zero_sun_dynamics, cr3bp_dynamics, sample_state):
    t_span = (0.0, 1.0)
    t_eval = np.linspace(*t_span, 21)
    bcr4bp = zero_sun_dynamics.propagate(sample_state, t_span, t_eval=t_eval)
    cr3bp = cr3bp_dynamics.propagate(sample_state, t_span, t_eval=t_eval)
    assert_allclose(bcr4bp["states"], cr3bp["states"], atol=1e-9)


def test_bcr4bp_jacobian_matches_finite_difference(bcr4bp_dynamics, sample_state):
    time = 0.7
    jacobian = bcr4bp_dynamics.compute_jacobian_A(time, sample_state)
    epsilon = 1e-7
    for column in range(6):
        plus = sample_state.copy()
        minus = sample_state.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        finite_difference = (
            bcr4bp_dynamics.equations_of_motion(time, plus)
            - bcr4bp_dynamics.equations_of_motion(time, minus)
        ) / (2.0 * epsilon)
        assert_allclose(jacobian[:, column], finite_difference, rtol=1e-6, atol=1e-8)


def test_solar_jacobian_block_matches_the_solar_acceleration_difference(
    bcr4bp_dynamics, sample_state
):
    time = 1.1
    jacobian = bcr4bp_dynamics.compute_jacobian_A(time, sample_state)
    hessian = pseudo_potential_hessian(Datum.DE421.mu, *sample_state[:3])
    solar_block = jacobian[3:, :3] - hessian
    epsilon = 1e-5

    for column in range(3):
        plus = sample_state[:3].copy()
        minus = sample_state[:3].copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        finite_difference = (
            bcr4bp_dynamics.sun_acceleration(time, plus)
            - bcr4bp_dynamics.sun_acceleration(time, minus)
        ) / (2.0 * epsilon)
        assert_allclose(solar_block[:, column], finite_difference, rtol=1e-6, atol=1e-10)


def test_bcr4bp_is_explicitly_time_dependent(bcr4bp_dynamics, sample_state):
    assert not np.allclose(
        bcr4bp_dynamics.equations_of_motion(0.0, sample_state),
        bcr4bp_dynamics.equations_of_motion(1.0, sample_state),
        atol=1e-10,
    )


def test_bcr4bp_stm_matches_a_small_initial_state_perturbation(bcr4bp_dynamics, sample_state):
    t_span = (0.0, 0.5)
    reference = bcr4bp_dynamics.propagate(sample_state, t_span, with_stm=True)
    delta = 1e-6
    direction = np.array([1.0, -2.0, 1.0, 0.5, -1.0, 2.0])
    direction /= np.linalg.norm(direction)
    perturbed = bcr4bp_dynamics.propagate(sample_state + delta * direction, t_span)

    assert_allclose(
        reference["stm"][-1] @ (delta * direction),
        perturbed["states"][-1] - reference["states"][-1],
        rtol=1e-4,
        atol=1e-9,
    )


def test_bcr4bp_rejects_jacobi_constant(bcr4bp_dynamics, sample_state):
    with pytest.raises(NotImplementedError):
        bcr4bp_dynamics.propagate(sample_state, (0.0, 0.5), with_jacobi=True)
    with pytest.raises(NotImplementedError):
        bcr4bp_dynamics.compute_jacobi_constant(sample_state)


@pytest.mark.spice
class TestBCR4BPEphemerisComparison:
    """双圆模型与含地月日点质量星历模型的短期能力对照。"""

    def test_short_propagation_stays_within_the_bicircular_model_error_scale(
        self,
        spice_manager,
        spice_eph_system,
        spice_syn_j2000,
        reference_epoch,
        sample_state,
    ):
        from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
        from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
        from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
        from e2m2e.algorithm.forces import ForceModel, PointMassGravity, ThirdBodyGravity

        et0 = spice_manager.utc_to_et(reference_epoch)
        spice_eph_system.coordinate_system = CoordinateSystem(
            axes=ICRSAxes(),
            origin=CelestialBodyOrigin(body="EARTH", spice=spice_manager),
        )
        solar_position = np.asarray(spice_eph_system.get_body_position("SUN", et0), dtype=float)
        solar_synodic = spice_syn_j2000.j2000_to_synodic(
            np.concatenate([solar_position, np.zeros(3)]), 0.0, et0
        )
        bcr4bp = BCR4BP_Dynamics(
            BCR4BPSystem.earth_moon(
                sun_phase0=float(np.arctan2(solar_synodic[1], solar_synodic[0]))
            )
        )
        cr3bp = CR3BP_Dynamics(spice_syn_j2000.cr3bp_system)
        initial_j2000 = spice_syn_j2000.synodic_to_j2000(sample_state, 0.0, et0)

        def ephemeris_final_state(days: float) -> np.ndarray:
            force_model = ForceModel(
                spice_eph_system,
                forces=[
                    PointMassGravity("EARTH"),
                    ThirdBodyGravity("MOON"),
                    ThirdBodyGravity("SUN"),
                ],
            )
            force_model.max_step = 600.0
            return np.asarray(
                force_model.propagate(
                    initial_j2000, (et0, et0 + days * SECONDS_PER_DAY), max_steps=1_000_000
                )["states"][-1]
            )

        def model_final_state(dynamics, days: float) -> np.ndarray:
            final_time = days * SECONDS_PER_DAY / bcr4bp.system.characteristic_time
            result = dynamics.propagate(sample_state, (0.0, final_time))
            return spice_syn_j2000.synodic_to_j2000(result["states"][-1], final_time, et0)

        ephemeris_one_day = ephemeris_final_state(1.0)
        bcr4bp_one_day_error = float(
            np.linalg.norm(model_final_state(bcr4bp, 1.0)[:3] - ephemeris_one_day[:3])
        )
        assert bcr4bp_one_day_error < 2_000.0

        ephemeris_two_days = ephemeris_final_state(2.0)
        bcr4bp_two_day_error = float(
            np.linalg.norm(model_final_state(bcr4bp, 2.0)[:3] - ephemeris_two_days[:3])
        )
        cr3bp_two_day_error = float(
            np.linalg.norm(model_final_state(cr3bp, 2.0)[:3] - ephemeris_two_days[:3])
        )
        assert bcr4bp_two_day_error < cr3bp_two_day_error
