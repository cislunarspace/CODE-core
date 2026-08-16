"""Rust 微分修正的能力边界与结果终止语义。"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


class NonCr3bpDynamics:
    """用于确认 Python 数值后端已移除的非 CR3BP 替身。"""

    system = object()


class ClosureOffsetDynamics:
    """只用于验证 Orbit 编排保留正确的闭合误差。"""

    def __init__(self, closure_offset: float) -> None:
        self.closure_offset = closure_offset
        self.system = object()

    def propagate(self, state, time_span, t_eval=None):
        del time_span
        initial_state = np.asarray(state, dtype=float)
        final_state = initial_state.copy()
        final_state[1] += self.closure_offset
        times = np.asarray([0.0, 1.0] if t_eval is None else [t_eval[0], t_eval[-1]])
        return {"time": times, "states": np.asarray([initial_state, final_state])}


def _initial_orbit() -> Orbit:
    orbit = Orbit(states=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], times=[0.0])
    orbit.period = 2.0
    return orbit


def test_iteration_requires_cr3bp_dynamics():
    corrector = DifferentialCorrection(NonCr3bpDynamics())
    corrector.setup_2D_symmetric_x_fixed_x0(x0=0.0)

    with pytest.raises(TypeError, match="CR3BP_Dynamics"):
        corrector.iterate_correction(_initial_orbit())


def test_symmetric_correction_reports_stagnation_when_update_is_too_small(
    dro_corrector, dro_seed_orbit
):
    dro_corrector.stagnation_limit = np.finfo(float).max

    result = dro_corrector.iterate_correction(dro_seed_orbit)

    assert result.status is ConvergenceState.STAGNATED
    assert result.cause is FailureCause.STAGNATION_DETECTED
    assert result.orbit is None


def test_full_period_correction_reports_stagnation_when_update_is_too_small(
    dro_corrector, dro_seed_orbit
):
    dro_corrector.setup_spo_fixed_x0(float(dro_seed_orbit.states[0, 0]))
    dro_corrector.stagnation_limit = np.finfo(float).max

    result = dro_corrector.iterate_full_period_correction(dro_seed_orbit)

    assert result.status is ConvergenceState.STAGNATED
    assert result.cause is FailureCause.STAGNATION_DETECTED
    assert result.orbit is None


def test_rust_result_exposes_convergence_history(dro_corrector, dro_seed_orbit):
    result = dro_corrector.iterate_correction(dro_seed_orbit)
    history = dro_corrector.get_convergence_history()

    assert result.status is ConvergenceState.CONVERGED
    assert result.cause is FailureCause.NONE
    assert len(history["errors"]) == result.iterations
    assert len(history["corrections"]) == result.iterations - 1
    assert result.residual_history[-1] == pytest.approx(result.residual)


def test_periodic_flag_uses_the_corrector_tolerance():
    corrector = DifferentialCorrection(ClosureOffsetDynamics(closure_offset=1e-9))
    corrector.setup_type = "spo_fixed_x0"
    corrector.tolerance = 1e-10

    orbit = corrector._create_corrected_orbit(
        {
            "state": np.zeros(6),
            "period": 2.0,
            "half_period": 1.0,
            "setup_type": corrector.setup_type,
            "error": None,
        }
    )

    assert orbit.closure_error == 1e-9
    assert orbit.is_periodic is False
