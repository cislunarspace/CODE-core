"""design_orbit 编排入口的最小真实调用冒烟。

本文件同时覆盖 LYAPUNOV CR3BP 初猜到 segmented 星历修正的真实路径。
"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import requires_spice

from e2m2e.algorithm.design import design_orbit
from e2m2e.algorithm.design.design_orbit import CORRECTION_TOL_KM
from e2m2e.data.templates import ConvergenceState
from tests.algorithm.design.conftest import make_design_request

pytestmark = [pytest.mark.orchestration, pytest.mark.spice, requires_spice]


def test_design_orbit_elfo_minimal_real_call():
    result = design_orbit(
        make_design_request(
            orbit_type="ELFO",
            semi_major_axis=3000.0,
            inclination=75.0,
            arg_of_pericenter=270.0,
            perilune_height=200.0,
            duration=14400.0,
            output_step=1200.0,
        )
    )
    assert result.orbit_type == "ELFO"
    assert result.cr3bp_orbit is None
    assert result.correction is None
    assert result.correction_method is None
    assert np.isnan(result.cr3bp_jacobi)
    assert result.initial_state.shape == (6,)
    assert len(result.ephemeris) > 0
    assert result.drift_e is not None
    assert result.moon_centric_elements is not None


@pytest.mark.time_budget(60)
def test_design_orbit_lyapunov_ephemeris_correction_smoke():
    result = design_orbit(
        make_design_request(
            orbit_type="LYAPUNOV",
            collinear_point=2,
            amplitude=12000.0,
            duration=1_270_000.0,
            output_step=7200.0,
        )
    )
    assert result.orbit_type == "LYAPUNOV"
    assert result.cr3bp_orbit is not None
    assert result.correction is not None
    assert result.correction.status is ConvergenceState.CONVERGED
    assert result.correction.max_residual < CORRECTION_TOL_KM
    assert result.correction_method == "segmented"
    assert len(result.ephemeris) > 0
