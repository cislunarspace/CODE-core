"""CR3BP 微分修正结果通过公开传播接口构造完整轨道。"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


def test_rust_result_builds_complete_orbit(corrected_dro):
    assert isinstance(corrected_dro, Orbit)
    assert corrected_dro.states.shape == (1000, 6)
    assert corrected_dro.times.shape == (1000,)
    assert corrected_dro.period > 0
    assert np.all(np.isfinite(corrected_dro.states))


def test_rust_result_keeps_status_contract(dro_corrector, dro_seed_orbit):
    result = dro_corrector.iterate_correction(dro_seed_orbit)

    assert result.status is ConvergenceState.CONVERGED
    assert result.cause is FailureCause.NONE
    assert result.message
    assert result.residual is not None
    assert result.residual < dro_corrector.tolerance


def test_non_cr3bp_correction_has_no_python_fallback():
    class NonCr3bpDynamics:
        system = object()

    corrector = DifferentialCorrection(NonCr3bpDynamics())
    corrector.setup_2D_symmetric_x_fixed_x0()
    guess = Orbit(states=np.zeros((1, 6)), times=[0.0])
    guess.period = 2.0

    with pytest.raises(TypeError, match="CR3BP_Dynamics"):
        corrector.iterate_correction(guess)
