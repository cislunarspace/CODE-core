"""DifferentialCorrection 全程走 dynamics.propagate() 集成测试。

验证有限差分 Jacobian 与 STM 一致、修正轨道构建、
以及 EphemerisDynamics 代码路径。
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.dynamics.dynamics import Dynamics
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.types.orbit import Orbit

# 地月系统质量比
MU = 1.21506683e-2

# 公共 fixtures 从 tests/algorithms/conftest.py 导入：
#   dro_dynamics, dro_corrector, dro_seed_orbit, corrected_dro


# ---------------------------------------------------------------------------
# CR3BP: finite-diff Jacobian 一致性
# ---------------------------------------------------------------------------


class TestFiniteDiffJacobianConsistency:
    """Refactor 后 _compute_jacobian_finite_diff 应与 STM 雅可比数值一致。"""

    def test_fd_jacobian_matches_stm(self, dro_dynamics):
        from tests.algorithm.conftest import DRO_VY0, DRO_X0

        state = np.array([DRO_X0, 0.0, 0.0, 0.0, DRO_VY0, 0.0])
        t_half = 3.153749

        corrector = DifferentialCorrection(dro_dynamics)
        corrector.setup_2D_symmetric_x_fixed_x0(x0=DRO_X0)

        fd_jac = corrector._compute_jacobian_finite_diff(state, t_half)

        # STM-based Jacobian
        result = dro_dynamics.propagate(state, (0, t_half), with_stm=True)
        stm = result["stm"][-1]
        deriv = dro_dynamics.equations_of_motion(t_half, result["states"][-1])

        n_constraints = len(corrector.constraint_indices)
        n_free = len(corrector.free_variable_indices)
        stm_jac = np.zeros((n_constraints, n_free))
        for j, var_idx in enumerate(corrector.free_variable_indices):
            if var_idx < 6:
                for i, c_idx in enumerate(corrector.constraint_indices):
                    stm_jac[i, j] = stm[c_idx, var_idx]
            elif var_idx == 6:
                for i, c_idx in enumerate(corrector.constraint_indices):
                    stm_jac[i, j] = deriv[c_idx]

        assert_allclose(fd_jac, stm_jac, atol=1e-4)


# ---------------------------------------------------------------------------
# CR3BP: _create_corrected_orbit 回归
# ---------------------------------------------------------------------------


class TestCreateCorrectedOrbitRegression:
    """_create_corrected_orbit 通过 propagate() 构建轨道，结果与之前一致。"""

    def test_returns_orbit(self, corrected_dro):
        assert isinstance(corrected_dro, Orbit)

    def test_states_shape(self, corrected_dro):
        assert corrected_dro.states.ndim == 2
        assert corrected_dro.states.shape[1] == 6
        assert corrected_dro.states.shape[0] == 1000

    def test_period(self, corrected_dro):
        assert corrected_dro.period > 0

    def test_closure(self, corrected_dro):
        err = np.linalg.norm(corrected_dro.states[-1] - corrected_dro.states[0])
        assert err < 1e-6


# ---------------------------------------------------------------------------
# 类型注解: Dynamics 基类
# ---------------------------------------------------------------------------


class TestDynamicsBaseClassAnnotation:
    """DifferentialCorrection 应接受 Dynamics 基类。"""

    def test_constructor_accepts_dynamics(self, dro_dynamics):
        corrector = DifferentialCorrection(dro_dynamics)
        assert isinstance(corrector.dynamics, Dynamics)


# ---------------------------------------------------------------------------
# SPICE-dependent: EphemerisDynamics 代码路径
# ---------------------------------------------------------------------------
# 公共 SPICE fixtures 来自 tests/conftest.py:
#   spice_manager, spice_eph_system, spice_eph_dynamics, spice_syn_j2000,
#   reference_epoch, spice_kernel_path

pytestmark = [pytest.mark.spice, pytest.mark.l3]


class TestEphemerisDynamicsCodePath:
    """验证 DifferentialCorrection 的 refactored 代码路径在 EphemerisDynamics 下工作。"""

    def test_create_corrected_orbit_with_ephemeris(self, spice_eph_dynamics):
        """_create_corrected_orbit 应通过 EphemerisDynamics.propagate() 构建 Orbit。"""
        r0 = 384400.0  # km (月球距离)
        v_circ = 1.018  # km/s (近似圆速度)
        initial_state = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0])
        half_period = np.pi * r0 / v_circ

        corrector = DifferentialCorrection(spice_eph_dynamics)
        corrector.setup_type = "2D_symmetric_x_fixed_x0"
        corrector.converged = True
        corrector.success = True
        corrector.current_error = 1e-8
        corrector.termination_reason = "test"
        corrector.tolerance = 1e-6

        result_dict = {
            "state": initial_state,
            "period": 2 * half_period,
            "half_period": half_period,
            "setup_type": corrector.setup_type,
            "converged": True,
            "error": 1e-8,
        }

        orbit = corrector._create_corrected_orbit(result_dict)

        assert isinstance(orbit, Orbit)
        assert orbit.states.shape[1] == 6
        assert orbit.period == pytest.approx(2 * half_period)
        assert np.all(np.isfinite(orbit.states))

    def test_fd_jacobian_with_ephemeris(self, spice_eph_dynamics):
        """_compute_jacobian_finite_diff 应通过 EphemerisDynamics.propagate() 计算。"""
        r0 = 384400.0
        v_circ = 1.018
        state = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0])
        t_half = np.pi * r0 / v_circ

        corrector = DifferentialCorrection(spice_eph_dynamics)
        corrector.constraint_indices = [1, 3]  # y, vx
        corrector.free_variable_indices = [4, 6]  # vy, T
        corrector.finite_difference_step = 1.0  # km 量级

        jac = corrector._compute_jacobian_finite_diff(state, t_half)

        assert jac.shape == (2, 2)
        assert np.all(np.isfinite(jac))

    def test_correction_converges_with_ephemeris(self, spice_eph_dynamics):
        """完整微分修正应在 EphemerisDynamics 下收敛（近圆轨道）。"""
        r0 = 384400.0
        v_circ = 1.018
        initial_state = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0])
        t_half = np.pi * r0 / v_circ

        guess = Orbit(states=[initial_state], times=[0])
        guess.period = 2 * t_half

        corrector = DifferentialCorrection(spice_eph_dynamics)
        corrector.setup_2D_symmetric_x_fixed_x0(x0=r0)
        corrector.tolerance = 1e-4

        result = corrector.iterate_correction(guess, verbose=False)

        assert result is not None, (
            f"EphemerisDynamics correction did not produce orbit. "
            f"termination: {corrector.termination_reason}"
        )
        assert isinstance(result, Orbit)
        assert corrector.converged, f"Correction did not converge: {corrector.termination_reason}"
