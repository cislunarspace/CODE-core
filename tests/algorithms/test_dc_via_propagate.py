"""Issue #31: DifferentialCorrection 全程走 dynamics.propagate() 集成测试

验证:
- _compute_jacobian_finite_diff 通过 propagate() 完成，结果与 STM 雅可比一致
- _create_corrected_orbit 通过 propagate() 构建轨道，返回 Orbit 对象
- DifferentialCorrection 可接受 Dynamics 基类（非 CR3BP_Dynamics 硬编码）
- (SPICE) EphemerisDynamics 跑通微分修正代码路径
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithms import DifferentialCorrection
from e2m2e.core import CR3BP_Dynamics, CR3BP_System, Orbit
from e2m2e.core.dynamics import Dynamics

MU = 1.21506683e-2


# ---------------------------------------------------------------------------
# CR3BP: finite-diff Jacobian 一致性
# ---------------------------------------------------------------------------


class TestFiniteDiffJacobianConsistency:
    """Refactor 后 _compute_jacobian_finite_diff 应与 STM 雅可比数值一致。"""

    @pytest.fixture
    def dynamics(self):
        return CR3BP_Dynamics(CR3BP_System(mu=MU, primary="earth", secondary="moon"))

    def test_fd_jacobian_matches_stm(self, dynamics):
        x0 = 0.79188556619742
        vy0 = 0.573665890385585
        state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        t_half = 3.153749

        corrector = DifferentialCorrection(dynamics)
        corrector.setup_2D_symmetric_x_fixed_x0(x0=x0)

        fd_jac = corrector._compute_jacobian_finite_diff(state, t_half)

        # STM-based Jacobian
        result = dynamics.propagate(state, (0, t_half), with_stm=True)
        stm = result["stm"][-1]
        deriv = dynamics.equations_of_motion(t_half, result["states"][-1])

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

    @pytest.fixture
    def corrected_orbit(self):
        system = CR3BP_System(mu=MU, primary="earth", secondary="moon")
        dynamics = CR3BP_Dynamics(system)

        x0 = 0.79188556619742
        guess = Orbit(states=[[x0, 0.0, 0.0, 0.0, 0.573665890385585, 0.0]], times=[0])
        guess.period = 6.307498

        corrector = DifferentialCorrection(dynamics)
        corrector.setup_2D_symmetric_x_fixed_x0(x0=x0)
        return corrector.iterate_correction(guess)

    def test_returns_orbit(self, corrected_orbit):
        assert isinstance(corrected_orbit, Orbit)

    def test_states_shape(self, corrected_orbit):
        assert corrected_orbit.states.ndim == 2
        assert corrected_orbit.states.shape[1] == 6
        assert corrected_orbit.states.shape[0] == 1000

    def test_period(self, corrected_orbit):
        assert corrected_orbit.period > 0

    def test_closure(self, corrected_orbit):
        err = np.linalg.norm(corrected_orbit.states[-1] - corrected_orbit.states[0])
        assert err < 1e-6


# ---------------------------------------------------------------------------
# 类型注解: Dynamics 基类
# ---------------------------------------------------------------------------


class TestDynamicsBaseClassAnnotation:
    """DifferentialCorrection 应接受 Dynamics 基类。"""

    def test_constructor_accepts_dynamics(self):
        system = CR3BP_System(mu=MU, primary="earth", secondary="moon")
        dynamics = CR3BP_Dynamics(system)
        corrector = DifferentialCorrection(dynamics)
        assert isinstance(corrector.dynamics, Dynamics)


# ---------------------------------------------------------------------------
# SPICE-dependent: EphemerisDynamics 代码路径
# ---------------------------------------------------------------------------

pytestmark_spice = pytest.mark.spice


def _make_eph_dynamics(spice_kernel_path):
    from e2m2e.core import EphemerisDynamics, EphemerisSystem, SPICEManager

    mgr = SPICEManager()
    mgr.load_kernel(spice_kernel_path)

    system = EphemerisSystem(
        bodies=["EARTH", "MOON", "SUN"],
        spice=mgr,
        origin="EARTH",
        frame="J2000",
    )
    dynamics = EphemerisDynamics(system)
    dynamics.rtol = 1e-10
    dynamics.atol = 1e-10
    dynamics.max_step = 600.0
    return dynamics, mgr


@pytest.fixture
def eph_dynamics(spice_kernel_path):
    dynamics, mgr = _make_eph_dynamics(spice_kernel_path)
    yield dynamics
    mgr.unload_kernel(spice_kernel_path)


class TestEphemerisDynamicsCodePath:
    """验证 DifferentialCorrection 的 refactored 代码路径在 EphemerisDynamics 下工作。"""

    def test_create_corrected_orbit_with_ephemeris(self, eph_dynamics):
        """_create_corrected_orbit 应通过 EphemerisDynamics.propagate() 构建 Orbit。"""
        r0 = 384400.0  # km (月球距离)
        v_circ = 1.018  # km/s (近似圆速度)
        initial_state = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0])
        half_period = np.pi * r0 / v_circ

        corrector = DifferentialCorrection(eph_dynamics)
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

    def test_fd_jacobian_with_ephemeris(self, eph_dynamics):
        """_compute_jacobian_finite_diff 应通过 EphemerisDynamics.propagate() 计算。"""
        r0 = 384400.0
        v_circ = 1.018
        state = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0])
        t_half = np.pi * r0 / v_circ

        corrector = DifferentialCorrection(eph_dynamics)
        corrector.constraint_indices = [1, 3]  # y, vx
        corrector.free_variable_indices = [4, 6]  # vy, T
        corrector.finite_difference_step = 1.0  # km 量级

        jac = corrector._compute_jacobian_finite_diff(state, t_half)

        assert jac.shape == (2, 2)
        assert np.all(np.isfinite(jac))

    def test_correction_converges_with_ephemeris(self, eph_dynamics):
        """完整微分修正应在 EphemerisDynamics 下收敛（近圆轨道）。"""
        r0 = 384400.0
        v_circ = 1.018
        initial_state = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0])
        t_half = np.pi * r0 / v_circ

        guess = Orbit(states=[initial_state], times=[0])
        guess.period = 2 * t_half

        corrector = DifferentialCorrection(eph_dynamics)
        corrector.setup_2D_symmetric_x_fixed_x0(x0=r0)
        corrector.tolerance = 1e-4

        result = corrector.iterate_correction(guess, verbose=False)

        assert result is not None, (
            f"EphemerisDynamics correction did not produce orbit. "
            f"termination: {corrector.termination_reason}"
        )
        assert isinstance(result, Orbit)
        assert corrector.converged, f"Correction did not converge: {corrector.termination_reason}"
