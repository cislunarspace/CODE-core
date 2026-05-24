"""
Unit tests for Orbit class
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from e2m2e.core.orbit import Orbit, OrbitFamily


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
        assert not orbit.is_periodic


class TestOrbitBasicProperties:
    """Tests for compute_basic_properties method"""

    def test_compute_properties_with_system(self, earth_moon_system):
        """Test compute_basic_properties with associated system"""
        states = np.random.rand(10, 6)
        times = np.linspace(0, 1, 10)
        orbit = Orbit(states=states, times=times, system=earth_moon_system)

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
        t = np.linspace(0, 1, 50)
        x = 0.5 + 0.1 * np.cos(2 * np.pi * t)
        states = np.column_stack(
            [x, np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50)]
        )
        orbit = Orbit(states=states, times=t)

        assert "x" in orbit.amplitudes
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
    """Tests for period property"""

    def test_period_returns_value(self):
        """Test period property returns value (auto-computed or None)"""
        states = np.random.rand(10, 6)
        times = np.linspace(0, 1, 10)
        orbit = Orbit(states=states, times=times)

        result = orbit.period
        assert result is None or isinstance(result, (float, np.floating))

    def test_period_with_estimate(self, sample_orbit):
        """Test period property returns period when estimated"""
        assert sample_orbit.period is None or isinstance(sample_orbit.period, (float, np.floating))


class TestOrbitAmplitude:
    """Tests for amplitudes property"""

    def test_amplitudes_valid_direction(self, sample_orbit):
        """Test amplitudes dict contains valid direction"""
        amp = sample_orbit.amplitudes["x"]
        assert isinstance(amp, float)
        assert amp >= 0

    def test_amplitudes_missing_direction(self, sample_orbit):
        """Test amplitudes dict raises KeyError for missing direction"""
        with pytest.raises(KeyError):
            _ = sample_orbit.amplitudes["invalid"]


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


class TestOrbitFamilyInit:
    """Tests for OrbitFamily initialization"""

    def test_orbit_family_rejects_non_orbit_list(self):
        """OrbitFamily should raise TypeError when given a list of non-Orbit objects"""
        with pytest.raises(TypeError, match="Orbit instances"):
            OrbitFamily(orbits=[1, 2, 3])

    def test_orbit_family_accepts_single_orbit(self, sample_orbit):
        """OrbitFamily should accept a single Orbit object"""
        family = OrbitFamily(orbits=sample_orbit)
        assert len(family) == 1


class TestOrbitPeriodicityCheck:
    """Tests for periodicity detection"""

    def test_periodicity_check(self):
        """Test periodicity detection with closed orbit data"""
        states = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        orbit = Orbit(states, times)

        assert orbit.period is not None


class TestOrbitFamilyType:
    """Tests for orbit family type"""

    def test_set_family_type(self):
        """Test setting orbit family type"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)
        orbit.family_type = "halo"

        assert orbit.family_type == "halo"

    def test_invalid_family_type(self):
        """Test family_type accepts arbitrary string values"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)
        orbit.family_type = "invalid_type"

        assert orbit.family_type == "invalid_type"
