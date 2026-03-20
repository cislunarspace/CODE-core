"""
CR3BP_Dynamics 类测试

测试动力学模型的核心功能，包括运动方程、轨迹传播、状态转移矩阵等。
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.core import CR3BP_System, CR3BP_Dynamics


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def earth_moon_system():
    """Create Earth-Moon CR3BP system."""
    return CR3BP_System.from_known_system("earth_moon")


@pytest.fixture
def dynamics(earth_moon_system):
    """Create CR3BP_Dynamics object."""
    return CR3BP_Dynamics(earth_moon_system)


@pytest.fixture
def sample_state():
    """Sample state vector near L1."""
    return np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])


# =============================================================================
# Test CR3BP_Dynamics Initialization
# =============================================================================
class TestDynamicsInit:
    """Test CR3BP_Dynamics initialization."""

    def test_init(self, earth_moon_system, dynamics):
        """Test basic initialization."""
        assert dynamics.system is earth_moon_system
        assert dynamics.integrator == "RK45"
        assert dynamics.rtol == 1e-12
        assert dynamics.atol == 1e-12
        assert dynamics.initialized is True

    def test_init_stores_system_reference(self, earth_moon_system, dynamics):
        """Test that dynamics object stores system reference."""
        assert dynamics.system.mu == earth_moon_system.mu

    def test_default_attributes(self, dynamics):
        """Test default attribute values."""
        assert dynamics.last_trajectory is None
        assert dynamics.last_stm is None
        assert dynamics.jacobi_history == []
        assert dynamics.jacobi_error == 0.0


# =============================================================================
# Test Equations of Motion
# =============================================================================
class TestEquationsOfMotion:
    """测试运动方程"""

    def test_equations_of_motion_shape(self, dynamics, sample_state):
        """Test that equations return 6 derivatives."""
        derivatives = dynamics.equations_of_motion(0.0, sample_state)
        assert derivatives.shape == (6,)

    def test_equations_of_motion_zeros(self, dynamics):
        """Test equations at equilibrium point."""
        # At Lagrange points with zero velocity, there should be specific behavior
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        derivatives = dynamics.equations_of_motion(0.0, state)

        # Velocity components should equal state velocity
        assert derivatives[0] == 0  # dx/dt = vx = 0
        assert derivatives[1] == 0  # dy/dt = vy = 0
        assert derivatives[2] == 0  # dz/dt = vz = 0

    def test_equations_of_motion_numerical_stability(self, dynamics, sample_state):
        """Test equations produce finite values."""
        derivatives = dynamics.equations_of_motion(0.0, sample_state)
        assert np.all(np.isfinite(derivatives))
        assert not np.any(np.isnan(derivatives))

    def test_equations_of_motion_velocity_input(self, dynamics):
        """Test that velocity components pass through correctly."""
        state = np.array([0.5, 0.0, 0.0, 1.0, 2.0, 3.0])
        derivatives = dynamics.equations_of_motion(0.0, state)

        # First three components are velocities
        assert derivatives[0] == 1.0
        assert derivatives[1] == 2.0
        assert derivatives[2] == 3.0


# =============================================================================
# Test STM Equations
# =============================================================================
class TestSTMEquations:
    """Test state transition matrix equations."""

    def test_stm_equations_shape(self, dynamics, sample_state):
        """Test that STM equations return 42 derivatives (6 + 36)."""
        augmented = np.concatenate([sample_state, np.eye(6).flatten()])
        derivatives = dynamics.equations_with_stm(0.0, augmented)
        assert derivatives.shape == (42,)

    def test_stm_equations_numerical_stability(self, dynamics, sample_state):
        """Test STM equations produce finite values."""
        augmented = np.concatenate([sample_state, np.eye(6).flatten()])
        derivatives = dynamics.equations_with_stm(0.0, augmented)
        assert np.all(np.isfinite(derivatives))


# =============================================================================
# Test Trajectory Propagation
# =============================================================================
class TestPropagate:
    """Test trajectory propagation."""

    def test_propagate_basic(self, dynamics, sample_state):
        """Test basic trajectory propagation."""
        result = dynamics.propagate(sample_state, [0, 1.0])

        assert "time" in result
        assert "states" in result
        assert "jacobi" in result
        assert "jacobi_error" in result
        assert len(result["states"]) > 0
        assert len(result["time"]) == len(result["states"])

    def test_propagate_with_stm(self, dynamics, sample_state):
        """Test propagation with state transition matrix."""
        result = dynamics.propagate(sample_state, [0, 1.0], with_stm=True)

        assert "stm" in result
        assert result["stm"].shape[1:] == (6, 6)
        assert len(result["stm"]) == len(result["states"])

    def test_propagate_stores_last_trajectory(self, dynamics, sample_state):
        """Test that propagation stores last trajectory."""
        dynamics.propagate(sample_state, [0, 1.0])
        assert dynamics.last_trajectory is not None
        assert len(dynamics.last_trajectory) == 2

    def test_propagate_jacobi_conservation(self, dynamics, sample_state):
        """Test that Jacobi constant is approximately conserved."""
        result = dynamics.propagate(sample_state, [0, 2.0])

        # Jacobi error should be small
        assert result["jacobi_error"] < 1e-4

    def test_propagate_custom_t_eval(self, dynamics, sample_state):
        """Test propagation with custom evaluation times."""
        t_eval = np.linspace(0, 1.0, 11)
        result = dynamics.propagate(sample_state, [0, 1.0], t_eval=t_eval)

        assert len(result["states"]) == len(t_eval)

    def test_propagate_stm_determinant(self, dynamics, sample_state):
        """Test that STM determinant is 1 (symplectic property)."""
        result = dynamics.propagate(sample_state, [0, 1.0], with_stm=True)

        for stm in result["stm"]:
            det = np.linalg.det(stm)
            assert_allclose(det, 1.0, rtol=1e-4)

    def test_propagate_returns_numpy_arrays(self, dynamics, sample_state):
        """Test that propagation returns numpy arrays."""
        result = dynamics.propagate(sample_state, [0, 1.0])

        assert isinstance(result["time"], np.ndarray)
        assert isinstance(result["states"], np.ndarray)


# =============================================================================
# Test State Transition Matrix
# =============================================================================
class TestStateTransitionMatrix:
    """Test STM computation."""

    def test_compute_stm_shape(self, dynamics, sample_state):
        """Test STM has correct shape."""
        stm = dynamics.compute_state_transition_matrix(sample_state, t=1.0)
        assert stm.shape == (6, 6)

    def test_compute_stm_determinant(self, dynamics, sample_state):
        """Test STM determinant is 1."""
        stm = dynamics.compute_state_transition_matrix(sample_state, t=1.0)
        det = np.linalg.det(stm)
        assert_allclose(det, 1.0, rtol=1e-4)

    def test_compute_stm_identity_at_zero(self, dynamics, sample_state):
        """Test STM is identity at t=0."""
        stm = dynamics.compute_state_transition_matrix(sample_state, t=0.0)
        assert_allclose(stm, np.eye(6), atol=1e-10)

    def test_compute_stm_positive_time(self, dynamics, sample_state):
        """Test STM at positive time is different from identity."""
        stm = dynamics.compute_state_transition_matrix(sample_state, t=0.5)
        # At small positive time, STM should be close to identity
        assert not np.allclose(stm, np.eye(6), atol=1e-3)


# =============================================================================
# Test Cross Section Detection
# =============================================================================
class TestCrossSection:
    """Test cross section detection."""

    def test_check_cross_section_x_plane(self, dynamics):
        """Test crossing detection on x-plane."""
        state_on = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_off = np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0])

        assert dynamics.check_cross_section(state_on, "x", 0.5) == True
        assert dynamics.check_cross_section(state_off, "x", 0.5) == False

    def test_check_cross_section_y_plane(self, dynamics):
        """Test crossing detection on y-plane."""
        state_on = np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
        state_off = np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0])

        assert dynamics.check_cross_section(state_on, "y", 0.3) == True
        assert dynamics.check_cross_section(state_off, "y", 0.3) == False

    def test_check_cross_section_z_plane(self, dynamics):
        """Test crossing detection on z-plane."""
        state_on = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
        state_off = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0])

        assert dynamics.check_cross_section(state_on, "z", 0.1) == True
        assert dynamics.check_cross_section(state_off, "z", 0.1) == False

    def test_check_cross_section_invalid_plane(self, dynamics, sample_state):
        """Test that invalid plane raises ValueError."""
        with pytest.raises(ValueError, match="无效的平面"):
            dynamics.check_cross_section(sample_state, "invalid", 0.0)


# =============================================================================
# Test Jacobi Constant
# =============================================================================
class TestComputeJacobiConstant:
    """Test Jacobi constant computation in dynamics."""

    def test_jacobi_constant(self, dynamics, sample_state):
        """Test Jacobi constant computation."""
        C = dynamics.compute_jacobi_constant(sample_state)
        assert isinstance(C, float)
        assert C > 0

    def test_jacobi_constant_at_lagrange_point(self, earth_moon_system, dynamics):
        """Test Jacobi constant at L1 point."""
        earth_moon_system.compute_libration_points()
        state = np.array([earth_moon_system.L1[0], 0, 0, 0, 0, 0])

        C = dynamics.compute_jacobi_constant(state)
        C_expected = earth_moon_system.get_jacobi_constant(state)
        assert_allclose(C, C_expected, rtol=1e-10)

    def test_jacobi_history_during_propagation(self, dynamics, sample_state):
        """Test that Jacobi history is recorded during propagation."""
        result = dynamics.propagate(sample_state, [0, 1.0])

        assert len(result["jacobi"]) > 0
        assert isinstance(result["jacobi"], list)


# =============================================================================
# Test String Representations
# =============================================================================
class TestStringRepr:
    """Test __str__ and __repr__ methods."""

    def test_str(self, dynamics):
        """Test __str__ method."""
        s = str(dynamics)
        assert "CR3BP_Dynamics" in s
        assert "RK45" in s

    def test_repr(self, dynamics):
        """Test __repr__ method."""
        r = repr(dynamics)
        assert "CR3BP_Dynamics" in r
        assert "rtol" in r
        assert "atol" in r


# =============================================================================
# Test Class Constants
# =============================================================================
class TestClassConstants:
    """Test class constants."""

    def test_stm_dimension(self):
        """Test STM dimension is 42 (6 state + 36 STM elements)."""
        assert CR3BP_Dynamics.STM_DIMENSION == 42

    def test_default_tolerance(self):
        """Test default tolerance value."""
        assert CR3BP_Dynamics.DEFAULT_TOLERANCE == 1e-12

    def test_default_max_step(self):
        """Test default max step value."""
        assert CR3BP_Dynamics.DEFAULT_MAX_STEP == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
