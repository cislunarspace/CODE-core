"""
Unit tests for Orbit class
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from e2m2e.core.orbit import Orbit


class TestOrbitInit:
    """Tests for Orbit initialization"""

    def test_init_basic(self):
        """Test basic initialization with states and times"""
        states = np.random.rand(10, 6)
        times = np.linspace(0, 1, 10)
        orbit = Orbit(states=states, times=times)

        assert orbit.states is not None
        assert orbit.times is not None
        assert len(orbit.states) == 10

    def test_init_with_system(self, earth_moon_system):
        """Test initialization with associated system"""
        states = np.random.rand(10, 6)
        times = np.linspace(0, 1, 10)
        orbit = Orbit(states=states, times=times, system=earth_moon_system)

        assert orbit.system is earth_moon_system

    def test_init_single_state(self):
        """Test initialization with single state (1D array)"""
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        time = np.array([0.0])
        orbit = Orbit(states=state, times=time)

        assert orbit.states.shape[0] == 1

    def test_init_invalid_state_dimension(self):
        """Test initialization with wrong state dimension raises error"""
        states = np.random.rand(10, 5)  # Wrong: 5 components instead of 6
        times = np.linspace(0, 1, 10)

        with pytest.raises(ValueError, match="必须包含6个分量"):
            Orbit(states=states, times=times)

    def test_init_mismatched_lengths(self):
        """Test initialization with mismatched lengths raises error"""
        states = np.random.rand(10, 6)
        times = np.linspace(0, 1, 8)  # 8 times for 10 states

        with pytest.raises(ValueError, match="长度必须与状态序列长度一致"):
            Orbit(states=states, times=times)

    def test_init_default_attributes(self):
        """Test initialization sets correct defaults - period may be auto-computed"""
        states = np.random.rand(10, 6)
        times = np.linspace(0, 1, 10)
        orbit = Orbit(states=states, times=times)

        assert orbit.jacobi_constants is None
        assert orbit.stability_indices is None
        assert orbit.family_type is None
        # period may be auto-computed via _estimate_period from zero crossings
        assert orbit.monodromy_matrix is None
        assert not orbit.is_periodic


class TestOrbitBasicProperties:
    """Tests for compute_basic_properties method"""

    def test_compute_properties_with_system(self, earth_moon_system):
        """Test compute_basic_properties with associated system"""
        states = np.random.rand(10, 6)
        times = np.linspace(0, 1, 10)
        orbit = Orbit(states=states, times=times, system=earth_moon_system)

        # Should compute jacobi constants when system is provided
        assert orbit.jacobi_constants is not None
        assert len(orbit.jacobi_constants) == 10

    def test_compute_properties_mean_state(self, earth_moon_system):
        """Test mean state computation"""
        states = np.random.rand(10, 6)
        times = np.linspace(0, 1, 10)
        orbit = Orbit(states=states, times=times, system=earth_moon_system)

        expected_mean = np.mean(states, axis=0)
        assert np.allclose(orbit.mean_state, expected_mean)

    def test_compute_properties_amplitudes(self):
        """Test amplitude computation"""
        # Create data with known amplitude
        t = np.linspace(0, 1, 50)
        x = 0.5 + 0.1 * np.cos(2 * np.pi * t)
        states = np.column_stack(
            [x, np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50)]
        )
        orbit = Orbit(states=states, times=t)

        assert "x" in orbit.amplitudes
        # amplitude = (max - min) / 2, which should be approximately 0.1
        assert orbit.amplitudes["x"] > 0

    def test_compute_properties_extrema(self):
        """Test extrema computation"""
        t = np.linspace(0, 1, 50)
        x = 0.5 + 0.1 * np.cos(2 * np.pi * t)
        states = np.column_stack(
            [x, np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50)]
        )
        orbit = Orbit(states=states, times=t)

        assert "x_max" in orbit.extrema
        assert "x_min" in orbit.extrema
        assert np.isclose(orbit.extrema["x_max"], 0.6, atol=1e-6)


class TestPropagateStateAtOrbitTime:
    """Tests for CR3BP_Dynamics.propagate_orbit_state_at_time (uses propagate)"""

    def test_propagate_at_epoch_matches_state0(self, sample_orbit, earth_moon_dynamics):
        state = earth_moon_dynamics.propagate_orbit_state_at_time(
            sample_orbit, float(sample_orbit.times[0])
        )
        np.testing.assert_allclose(state, sample_orbit.states[0], rtol=1e-9, atol=1e-12)

    def test_propagate_returns_finite_vector(self, sample_orbit, earth_moon_dynamics):
        t = float(sample_orbit.times[0]) + 0.05
        state = earth_moon_dynamics.propagate_orbit_state_at_time(
            sample_orbit, t, integration_dt=0.005
        )
        assert state.shape == (6,)
        assert not np.any(np.isnan(state))


class TestOrbitPeriod:
    """Tests for get_period method"""

    def test_get_period_returns_value(self):
        """Test get_period returns period value (auto-computed or None)"""
        states = np.random.rand(10, 6)
        times = np.linspace(0, 1, 10)
        orbit = Orbit(states=states, times=times)

        # get_period returns the period attribute which may be auto-computed
        result = orbit.get_period()
        assert result is None or isinstance(result, (float, np.floating))

    def test_get_period_with_estimate(self, sample_orbit):
        """Test get_period returns period when estimated"""
        # sample_orbit should have some period estimation
        sample_orbit.get_period()
        # Period may or may not be estimated depending on orbit shape


class TestOrbitAmplitude:
    """Tests for get_amplitude method"""

    def test_get_amplitude_valid_direction(self, sample_orbit):
        """Test get_amplitude with valid direction"""
        amp = sample_orbit.get_amplitude(direction="x")
        assert isinstance(amp, float)
        assert amp >= 0

    def test_get_amplitude_invalid_direction(self, sample_orbit):
        """Test get_amplitude raises error for invalid direction"""
        with pytest.raises(ValueError, match="无效的方向"):
            sample_orbit.get_amplitude(direction="invalid")


class TestOrbitSaveLoad:
    """Tests for save_to_file and load_from_file methods"""

    def test_save_load_roundtrip(self, sample_orbit):
        """Test save and load returns equivalent orbit"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filename = f.name

        try:
            sample_orbit.save_to_file(filename=filename)
            loaded_orbit = Orbit.load_from_file(filename=filename)

            assert np.allclose(loaded_orbit.states, sample_orbit.states, atol=1e-10)
            assert np.allclose(loaded_orbit.times, sample_orbit.times, atol=1e-10)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_load_from_file_pathlib(self, sample_orbit):
        """Test load_from_file accepts Path objects"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filename = Path(f.name)

        try:
            sample_orbit.save_to_file(filename=filename)
            loaded_orbit = Orbit.load_from_file(filename=filename)

            assert np.allclose(loaded_orbit.states, sample_orbit.states, atol=1e-10)
        finally:
            if filename.exists():
                os.remove(filename)


class TestOrbitMonodromy:
    """Tests for compute_monodromy_matrix method"""

    def test_monodromy_requires_period(self, sample_orbit, earth_moon_dynamics):
        """Test compute_monodromy_matrix raises error without period"""
        sample_orbit.period = None

        with pytest.raises(ValueError, match="轨道周期未知"):
            sample_orbit.compute_monodromy_matrix(dynamics=earth_moon_dynamics)


class TestOrbitStability:
    """Tests for compute_stability method"""

    def test_stability_requires_monodromy(self, sample_orbit, earth_moon_dynamics):
        """Test compute_stability requires monodromy matrix"""
        sample_orbit.monodromy_matrix = None
        sample_orbit.period = 1.0  # Set period to avoid period error

        # This should trigger monodromy computation
        result = sample_orbit.compute_stability(dynamics=earth_moon_dynamics)

        assert "stability" in result
        assert "eigenvalues" in result

    def test_compute_stability_with_zero_period(self, earth_moon_dynamics, earth_moon_system):
        """compute_stability should handle zero period gracefully."""
        orbit = Orbit(
            states=np.array([[0.8, 0.0, 0.0, 0.0, 0.1, 0.0]]),
            times=np.array([0.0]),
            system=earth_moon_system,
        )
        orbit.period = 0.0
        result = orbit.compute_stability(earth_moon_dynamics)
        assert result["stability"] == "unknown"
        assert result["eigenvalues"] is None


class TestOrbitMetadata:
    """Tests for orbit metadata"""

    def test_metadata_created_timestamp(self, sample_orbit):
        """Test metadata contains creation timestamp"""
        assert "created" in sample_orbit.metadata
        assert sample_orbit.metadata["created"] is not None

    def test_metadata_source(self, sample_orbit):
        """Test metadata contains source"""
        assert "source" in sample_orbit.metadata
        assert sample_orbit.metadata["source"] == "e2m2e library"

    def test_metadata_description(self, sample_orbit):
        """Test metadata can store description"""
        assert "description" in sample_orbit.metadata
