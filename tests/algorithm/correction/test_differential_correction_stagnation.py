"""微分修正的停滞与周期标记回归测试。"""

from __future__ import annotations

import numpy as np

from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import Orbit


class StagnantDynamics:
    """返回固定残差且零修正量的确定性动力学替身。"""

    def __init__(self, closure_offset: float = 1e-9) -> None:
        self.closure_offset = closure_offset
        self.system = object()

    def propagate(self, state, time_span, t_eval=None, with_stm=False, with_jacobi=False):
        del time_span, with_jacobi
        initial_state = np.array(state, dtype=float)
        final_state = initial_state.copy()
        final_state[1] += self.closure_offset
        result = {
            "time": np.array([0.0, 1.0]) if t_eval is None else np.array([t_eval[0], t_eval[-1]]),
            "states": np.array([initial_state, final_state]),
        }
        if with_stm:
            final_stm = np.eye(6)
            final_stm[1, 1] = 1e6 + 1
            final_stm[3, 3] = 1e6 + 1
            final_stm[4, 4] = 1e6 + 1
            final_stm[1, 4] = 1e6
            result["stm"] = np.array([np.eye(6), final_stm])
        return result

    def equations_of_motion(self, time, state):
        del time, state
        derivative = np.zeros(6)
        derivative[3] = 1e6
        return derivative


def _initial_orbit() -> Orbit:
    orbit = Orbit(states=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], times=[0.0])
    orbit.period = 2.0
    return orbit


def test_symmetric_correction_reports_stagnation_when_small_residual_cannot_be_reduced():
    corrector = DifferentialCorrection(StagnantDynamics())
    corrector.setup_2D_symmetric_x_fixed_x0(x0=0.0)

    result = corrector.iterate_correction(_initial_orbit())

    assert result.status is ConvergenceState.STAGNATED
    assert result.cause is FailureCause.STAGNATION_DETECTED
    assert result.orbit is None


def test_full_period_correction_reports_stagnation_when_small_residual_cannot_be_reduced():
    corrector = DifferentialCorrection(StagnantDynamics())
    corrector.setup_spo_fixed_x0(x0=0.0)

    result = corrector.iterate_full_period_correction(_initial_orbit())

    assert result.status is ConvergenceState.STAGNATED
    assert result.cause is FailureCause.STAGNATION_DETECTED
    assert result.orbit is None


def test_periodic_flag_uses_the_corrector_tolerance():
    corrector = DifferentialCorrection(StagnantDynamics(closure_offset=1e-9))
    corrector.setup_spo_fixed_x0(x0=0.0)
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
