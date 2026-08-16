"""平面三角平动点全周期 PAL 的族生成接缝测试。"""

import numpy as np
import pytest

from e2m2e.algorithm.family.lpo_initial_guess import compute_lpo_initial_guess
from e2m2e.algorithm.family.planar_continuation import generate_planar_periodic_family
from e2m2e.algorithm.family.spo_initial_guess import compute_spo_initial_guess
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import Orbit
from e2m2e.integrators import planar_full_period_pal_py


def _corrected_planar_seed(dynamics, family_type: str, point: int) -> Orbit:
    if family_type == "spo":
        state, period = compute_spo_initial_guess(dynamics.system, point, amplitude_km=1000.0)
        setup = DifferentialCorrection.setup_spo_fixed_x0
    else:
        state, period = compute_lpo_initial_guess(dynamics.system, point, amplitude_km=1000.0)
        setup = DifferentialCorrection.setup_lpo_fixed_x0
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = period
    corrector = DifferentialCorrection(dynamics)
    setup(corrector, float(state[0]), point)
    result = corrector.iterate_full_period_correction(seed, verbose=False)
    assert result.orbit is not None, result.message
    return result.orbit


@pytest.mark.parametrize(("family_type", "point"), [("spo", 4), ("spo", 5), ("lpo", 4), ("lpo", 5)])
def test_generates_planar_family_with_full_period_closure(
    earth_moon_dynamics, family_type: str, point: int
):
    """SPO/LPO 种子经公开族生成接缝返回连续的完整周期成员。"""
    result = generate_planar_periodic_family(
        earth_moon_dynamics,
        _corrected_planar_seed(earth_moon_dynamics, family_type, point),
        family_type=family_type,
        libration_point=point,
        n_orbits=2,
        step_size=0.01,
        initial_direction="decrease-x0",
    )

    assert result.status is ConvergenceState.CONVERGED, result.message
    assert result.steps == 2
    assert len(result.family) == 3
    assert result.family.family_type == family_type
    assert result.family.system is earth_moon_dynamics.system

    for orbit in result.family:
        assert orbit.period is not None
        assert orbit.closure_error is not None
        assert orbit.closure_error <= 1e-8
        assert np.allclose(orbit.states[0, (2, 5)], 0.0, atol=1e-12)
        propagated = earth_moon_dynamics.propagate(
            orbit.states[0],
            (0.0, orbit.period),
            t_eval=np.linspace(0.0, orbit.period, 64),
            with_jacobi=True,
        )
        jacobi = np.asarray(propagated["jacobi"])
        assert np.max(np.abs(jacobi - jacobi[0])) <= 1e-10
        assert orbit.parameters["jacobi_drift"] <= 1e-10

    assert result.family.metadata["planar_pal"] == "rust-full-period-pal"


def test_l4_spo_increase_x0_direction_keeps_full_period_closure(earth_moon_dynamics):
    """反向初始切向量使用同一族生成接缝，并满足同样的完整闭合契约。"""
    result = generate_planar_periodic_family(
        earth_moon_dynamics,
        _corrected_planar_seed(earth_moon_dynamics, "spo", 4),
        family_type="spo",
        libration_point=4,
        n_orbits=1,
        step_size=0.01,
        initial_direction="increase-x0",
    )

    assert result.status is ConvergenceState.CONVERGED, result.message
    assert len(result.family) == 2
    assert result.family[-1].period is not None
    assert result.family[-1].closure_error <= 1e-8


def test_lpo_continuation_crosses_period_turning_region(earth_moon_dynamics):
    """LPO 长程链允许周期转向，族身份由弧长而不是周期单调性表达。"""
    result = generate_planar_periodic_family(
        earth_moon_dynamics,
        _corrected_planar_seed(earth_moon_dynamics, "lpo", 4),
        family_type="lpo",
        libration_point=4,
        n_orbits=12,
        step_size=0.01,
        initial_direction="decrease-x0",
    )

    assert result.status is ConvergenceState.CONVERGED, result.message
    periods = np.array([orbit.period for orbit in result.family])
    assert len(periods) == 13
    assert np.all(np.isfinite(periods))
    assert np.any(np.diff(periods) > 0.0)
    assert np.any(np.diff(periods) < 0.0)


def test_iteration_limit_preserves_seed_family(earth_moon_dynamics):
    """内核软失败时仍返回带种子的部分族与稳定状态三元组。"""
    seed = _corrected_planar_seed(earth_moon_dynamics, "spo", 4)
    raw = planar_full_period_pal_py(
        earth_moon_dynamics.system.mu,
        seed.states[0].tolist(),
        float(seed.period),
        1,
        0.01,
        "decrease-x0",
        max_iterations=0,
        rtol=earth_moon_dynamics.rtol,
        atol=earth_moon_dynamics.atol,
        max_step=earth_moon_dynamics.max_step,
    )

    assert raw.status is ConvergenceState.MAX_ITERATIONS
    assert raw.cause is FailureCause.MAX_ITERATIONS_REACHED
    assert len(raw.states) == 1
    assert raw.steps == 0


def test_rejects_nonplanar_seed(earth_moon_dynamics):
    """族生成接缝拒绝非平面种子，不把三维问题传入平面 Rust 内核。"""
    seed = _corrected_planar_seed(earth_moon_dynamics, "spo", 4)
    seed.states[0, 2] = 1e-5

    with pytest.raises(ValueError, match="z=vz=0"):
        generate_planar_periodic_family(
            earth_moon_dynamics,
            seed,
            family_type="spo",
            libration_point=4,
            n_orbits=1,
            step_size=0.01,
            initial_direction="decrease-x0",
        )


def test_rejects_unknown_initial_direction(earth_moon_dynamics):
    """初始方向是受控接缝值，未知值直接拒绝。"""
    seed = _corrected_planar_seed(earth_moon_dynamics, "spo", 4)

    with pytest.raises(ValueError, match="initial_direction"):
        generate_planar_periodic_family(
            earth_moon_dynamics,
            seed,
            family_type="spo",
            libration_point=4,
            n_orbits=1,
            step_size=0.01,
            initial_direction="sideways",
        )
