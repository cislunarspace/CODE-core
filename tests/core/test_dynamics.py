"""
Unit tests for CR3BP_Dynamics class
"""

import pytest
import numpy as np
from e2m2e.core.dynamics import CR3BP_Dynamics


class TestCR3BPDynamicsInit:
    """Tests for CR3BP_Dynamics initialization"""

    def test_init_attributes(self, earth_moon_system):
        """Test initialization sets correct attributes"""
        dynamics = CR3BP_Dynamics(system=earth_moon_system)
        assert dynamics.system is earth_moon_system
        assert dynamics.integrator == "RK45"
        assert dynamics.rtol == 1e-12
        assert dynamics.atol == 1e-12
        assert dynamics.max_step == 0.01
        assert dynamics.initialized is True

    def test_init_stores_system(self, earth_moon_system):
        """Test dynamics stores reference to system"""
        dynamics = CR3BP_Dynamics(system=earth_moon_system)
        assert dynamics.system.mu == earth_moon_system.mu


class TestCR3BPDynamicsEquationsOfMotion:
    """Tests for equations_of_motion method"""

    def test_equations_returns_six_components(self, earth_moon_dynamics):
        """Test equations return 6 component derivative vector"""
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        derivative = earth_moon_dynamics.equations_of_motion(t=0.0, state=state)
        
        assert len(derivative) == 6

    def test_equations_velocity_components(self, earth_moon_dynamics):
        """Test first three components are velocities"""
        state = np.array([0.8, 0.0, 0.0, 1.0, 2.0, 3.0])
        derivative = earth_moon_dynamics.equations_of_motion(t=0.0, state=state)
        
        assert derivative[0] == 1.0  # vx
        assert derivative[1] == 2.0  # vy
        assert derivative[2] == 3.0  # vz

    def test_equations_z_plane_motion(self, earth_moon_dynamics):
        """Test motion in z=0 plane"""
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        derivative = earth_moon_dynamics.equations_of_motion(t=0.0, state=state)
        
        # z acceleration should be zero when z=0
        assert derivative[2] == 0.0  # vz dot
        assert derivative[5] == 0.0  # az

    def test_equations_symmetry(self, earth_moon_dynamics):
        """Test symmetry of equations at y=0"""
        state1 = np.array([0.8, 0.1, 0.0, 0.0, 0.1, 0.0])
        state2 = np.array([0.8, -0.1, 0.0, 0.0, 0.1, 0.0])
        
        deriv1 = earth_moon_dynamics.equations_of_motion(t=0.0, state=state1)
        deriv2 = earth_moon_dynamics.equations_of_motion(t=0.0, state=state2)
        
        # x and z accelerations should be same for symmetric y states
        assert deriv1[0] == deriv2[0]
        assert deriv1[2] == deriv2[2]


class TestCR3BPDynamicsEquationsWithSTM:
    """Tests for equations_with_stm method"""

    def test_stm_equations_length(self, earth_moon_dynamics):
        """Test augmented state returns 42 components"""
        augmented_state = np.zeros(42)
        augmented_state[:6] = [0.8, 0.0, 0.0, 0.0, 0.1, 0.0]
        
        derivative = earth_moon_dynamics.equations_with_stm(t=0.0, augmented_state=augmented_state)
        
        assert len(derivative) == 42

    def test_stm_equations_first_six(self, earth_moon_dynamics):
        """Test first 6 components match regular equations"""
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        augmented_state = np.concatenate([state, np.eye(6).flatten()])
        
        deriv_regular = earth_moon_dynamics.equations_of_motion(t=0.0, state=state)
        deriv_augmented = earth_moon_dynamics.equations_with_stm(t=0.0, augmented_state=augmented_state)
        
        assert np.allclose(deriv_regular, deriv_augmented[:6])


class TestCR3BPDynamicsPropagate:
    """Tests for propagate method"""

    def test_propagate_returns_dict(self, earth_moon_dynamics, sample_state):
        """Test propagate returns expected dictionary structure"""
        result = earth_moon_dynamics.propagate(
            initial_state=sample_state,
            t_span=[0, 1.0],
        )

        assert isinstance(result, dict)
        assert "time" in result
        assert "states" in result
        assert "jacobi" not in result
        assert "jacobi_error" not in result

    def test_propagate_time_points(self, earth_moon_dynamics, sample_state):
        """Test propagate with specific evaluation times"""
        t_eval = np.linspace(0, 1.0, 10)
        result = earth_moon_dynamics.propagate(
            initial_state=sample_state,
            t_span=[0, 1.0],
            t_eval=t_eval
        )
        
        assert len(result["time"]) == len(t_eval)
        assert len(result["states"]) == len(t_eval)

    def test_propagate_stores_trajectory(self, earth_moon_dynamics, sample_state):
        """Test propagate stores trajectory in object"""
        earth_moon_dynamics.propagate(
            initial_state=sample_state,
            t_span=[0, 1.0]
        )
        
        assert earth_moon_dynamics.last_trajectory is not None
        assert len(earth_moon_dynamics.last_trajectory) == 2

    def test_propagate_with_stm(self, earth_moon_dynamics, sample_state):
        """Test propagate with STM computation"""
        result = earth_moon_dynamics.propagate(
            initial_state=sample_state,
            t_span=[0, 1.0],
            with_stm=True
        )
        
        assert "stm" in result
        assert result["stm"].shape == (len(result["states"]), 6, 6)

    def test_propagate_stm_determinant(self, earth_moon_dynamics, sample_state):
        """Test STM has determinant of 1 (symplectic property)"""
        result = earth_moon_dynamics.propagate(
            initial_state=sample_state,
            t_span=[0, 0.5],
            with_stm=True
        )
        
        final_stm = result["stm"][-1]
        det = np.linalg.det(final_stm)
        assert np.isclose(det, 1.0, atol=1e-10)

    def test_propagate_jacobi_constant_conservation(self, earth_moon_dynamics, sample_state):
        """Test Jacobi constant is approximately conserved"""
        result = earth_moon_dynamics.propagate(
            initial_state=sample_state,
            t_span=[0, 2.0],
            with_jacobi=True,
        )

        # Jacobi constant should be nearly constant
        jacobi_variation = np.max(np.abs(np.diff(result["jacobi"])))
        assert jacobi_variation < 1e-6

    def test_propagate_jacobi_error_stored(self, earth_moon_dynamics, sample_state):
        """Test jacobi_error is computed and stored"""
        result = earth_moon_dynamics.propagate(
            initial_state=sample_state,
            t_span=[0, 1.0],
            with_jacobi=True,
        )

        assert isinstance(result["jacobi_error"], float)
        assert result["jacobi_error"] >= 0


class TestCR3BPDynamicsStateTransitionMatrix:
    """Tests for compute_state_transition_matrix method"""

    def test_stm_shape(self, earth_moon_dynamics, sample_state):
        """Test STM is 6x6 matrix"""
        stm = earth_moon_dynamics.compute_state_transition_matrix(
            initial_state=sample_state,
            t=1.0
        )
        
        assert stm.shape == (6, 6)

    def test_stm_initial_identity(self, earth_moon_dynamics, sample_state):
        """Test STM at t=0 is identity"""
        stm = earth_moon_dynamics.compute_state_transition_matrix(
            initial_state=sample_state,
            t=0.0
        )
        
        assert np.allclose(stm, np.eye(6), atol=1e-10)

    def test_stm_determinant_unity(self, earth_moon_dynamics, sample_state):
        """Test STM determinant is 1"""
        stm = earth_moon_dynamics.compute_state_transition_matrix(
            initial_state=sample_state,
            t=0.5
        )
        
        assert np.isclose(np.linalg.det(stm), 1.0, atol=1e-10)


class TestCR3BPDynamicsJacobiConstant:
    """Tests for compute_jacobi_constant method"""

    def test_jacobi_constant_computation(self, earth_moon_dynamics, earth_moon_system):
        """Test Jacobi constant matches system calculation"""
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        
        jacobi_from_dynamics = earth_moon_dynamics.compute_jacobi_constant(state=state)
        jacobi_from_system = earth_moon_system.get_jacobi_constant(state=state)
        
        assert np.isclose(jacobi_from_dynamics, jacobi_from_system)


class TestCR3BPDynamicsCrossSection:
    """Tests for check_cross_section method"""

    def test_cross_section_returns_bool(self, earth_moon_dynamics):
        """Test check_cross_section returns boolean"""
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.1, 0.0])
        result = earth_moon_dynamics.check_cross_section(
            state=state,
            plane="x",
            value=0.5
        )
        
        assert isinstance(result, (bool, np.bool_))

    def test_cross_section_detects_crossing(self, earth_moon_dynamics):
        """Test cross section detects when state crosses plane"""
        # State at x=0.5 should cross x=0.5 plane
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.1, 0.0])
        result = earth_moon_dynamics.check_cross_section(
            state=state,
            plane="x",
            value=0.5
        )
        
        assert result == True

    def test_cross_section_detects_no_crossing(self, earth_moon_dynamics):
        """Test cross section detects when state doesn't cross"""
        # State at x=0.8 doesn't cross x=0.5 plane
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        result = earth_moon_dynamics.check_cross_section(
            state=state,
            plane="x",
            value=0.5
        )
        
        assert result == False

    def test_cross_section_z_plane(self, earth_moon_dynamics):
        """Test z-plane cross section"""
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.1, 0.0])
        result = earth_moon_dynamics.check_cross_section(
            state=state,
            plane="z",
            value=0.0
        )
        
        assert result == True
