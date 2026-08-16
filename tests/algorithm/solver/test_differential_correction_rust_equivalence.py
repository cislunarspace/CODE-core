"""CR3BP 微分修正的 Rust 数值内核验收。"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.templates import ConvergenceState
from e2m2e.data.types.orbit import Orbit
from e2m2e.integrators import differential_correction_cr3bp_py

pytestmark = pytest.mark.orchestration


def test_default_correction_converges_with_rust_core(dro_dynamics, dro_seed_orbit):
    """公开入口应以 Rust 内核收敛到确定的 DRO 初态。"""
    corrector = DifferentialCorrection(dro_dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0=float(dro_seed_orbit.states[0, 0]))

    result = corrector.iterate_correction(dro_seed_orbit)

    assert result.status is ConvergenceState.CONVERGED
    assert result.orbit is not None
    assert result.orbit.states[0, 0] == pytest.approx(dro_seed_orbit.states[0, 0])
    np.testing.assert_allclose(result.orbit.states[0, [1, 2, 3, 5]], 0.0, atol=1e-10)


def test_xz_symmetric_halo_converges_with_rust_core(cr3bp_dynamics):
    """XZ 对称（Halo 固定 z0）配置应经 Rust 内核收敛。"""
    from e2m2e.algorithm.family.halo_initial_guess import compute_halo_initial_guess

    z0 = 0.001
    guess = compute_halo_initial_guess(
        mu=float(cr3bp_dynamics.system.mu),
        z_amplitude=z0,
        L=1,
        halo_class=0,
    )
    state = np.array([guess["x0"], 0.0, z0, 0.0, guess["vy0"], 0.0])
    seed = Orbit(states=[state], times=[0.0])
    seed.period = 2.0 * guess["T_half"]

    corrector = DifferentialCorrection(cr3bp_dynamics)
    corrector.setup_halo_orbit_fixed_z0(z0, 1)

    result = corrector.iterate_correction(seed)

    assert result.status is ConvergenceState.CONVERGED
    assert result.orbit is not None
    assert result.orbit.states[0, 2] == pytest.approx(z0)
    np.testing.assert_allclose(result.orbit.states[0, [1, 3, 5]], 0.0, atol=1e-10)


def test_rust_ffi_rejects_out_of_range_free_variable():
    """Rust FFI 不应静默忽略非法自由变量索引。"""
    with pytest.raises(ValueError, match="free_variable_indices"):
        differential_correction_cr3bp_py(
            mu=0.0121505856,
            initial_state=[0.8, 0.0, 0.0, 0.0, 0.2, 0.0],
            initial_time=0.1,
            constraint_indices=[1],
            target_values=[0.0],
            free_variable_indices=[7],
            full_period=False,
            recover_halo_time=False,
            max_iterations=1,
            tolerance=1e-12,
            stagnation_limit=1e-14,
            divergence_limit=1e10,
            rtol=1e-12,
            atol=1e-12,
        )


def test_python_backend_parameter_is_not_available(dro_dynamics, dro_seed_orbit):
    """Python 数值后端已移除，公开迭代接口不再接受 backend 参数。"""
    corrector = DifferentialCorrection(dro_dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0=float(dro_seed_orbit.states[0, 0]))

    with pytest.raises(TypeError, match="backend"):
        corrector.iterate_correction(dro_seed_orbit, backend="python")
