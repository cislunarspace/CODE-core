"""
Unit tests for CR3BP_System class
"""

import numpy as np
import pytest

from e2m2e.core.system import CR3BP_System, LibrationPoint


class TestLibrationPointEnum:
    """Tests for LibrationPoint enum"""

    def test_libration_point_values(self):
        """Test all five libration points exist"""
        assert LibrationPoint.L1.value == 1
        assert LibrationPoint.L2.value == 2
        assert LibrationPoint.L3.value == 3
        assert LibrationPoint.L4.value == 4
        assert LibrationPoint.L5.value == 5

    def test_libration_point_count(self):
        """Test there are exactly 5 libration points"""
        assert len(LibrationPoint) == 5


class TestCR3BPSystemInit:
    """Tests for CR3BP_System initialization"""

    def test_init_basic_attributes(self):
        """Test basic initialization sets attributes correctly"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        assert system.mu == 0.01215
        assert system.primary_body == "Earth"
        assert system.secondary_body == "Moon"

    def test_init_defaults(self):
        """Test initialization sets correct defaults"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        assert system.L_points is None
        assert system.L1 is None
        assert system.has_L_points is False
        assert system.is_initialized is False
        assert system.characteristic_length is None
        assert system.characteristic_time is None
        assert system.characteristic_velocity is None


class TestCR3BPSystemKnownSystems:
    """Tests for from_known_system class method"""

    def test_earth_moon_system(self, earth_moon_system):
        """Test Earth-Moon system creation"""
        assert earth_moon_system.mu == 0.01215
        assert earth_moon_system.primary_body == "Earth"
        assert earth_moon_system.secondary_body == "Moon"

    def test_sun_earth_system(self, sun_earth_system):
        """Test Sun-Earth system creation"""
        assert sun_earth_system.mu == 3.0039e-6
        assert sun_earth_system.primary_body == "Sun"
        assert sun_earth_system.secondary_body == "Earth"

    def test_sun_jupiter_system(self, sun_jupiter_system):
        """Test Sun-Jupiter system creation"""
        assert sun_jupiter_system.mu == 0.0009535
        assert sun_jupiter_system.primary_body == "Sun"
        assert sun_jupiter_system.secondary_body == "Jupiter"

    def test_invalid_system_name(self):
        """Test ValueError for unknown system name"""
        with pytest.raises(ValueError, match="未知系统"):
            CR3BP_System.from_known_system("invalid_system")

    def test_known_systems_keys(self):
        """Test KNOWN_SYSTEMS contains expected systems"""
        assert "earth_moon" in CR3BP_System.KNOWN_SYSTEMS
        assert "sun_earth" in CR3BP_System.KNOWN_SYSTEMS
        assert "sun_jupiter" in CR3BP_System.KNOWN_SYSTEMS


class TestCR3BPSystemLibrationPoints:
    """Tests for compute_libration_points method"""

    def test_compute_libration_points(self, earth_moon_system):
        """Test libration points computation"""
        L_points = earth_moon_system.compute_libration_points()

        assert earth_moon_system.has_L_points is True
        assert len(L_points) == 5
        assert LibrationPoint.L1 in L_points
        assert LibrationPoint.L2 in L_points
        assert LibrationPoint.L3 in L_points
        assert LibrationPoint.L4 in L_points
        assert LibrationPoint.L5 in L_points

    def test_libration_points_stored(self, earth_moon_system):
        """Test libration points are stored in system"""
        earth_moon_system.compute_libration_points()

        assert earth_moon_system.L1 is not None
        assert earth_moon_system.L2 is not None
        assert earth_moon_system.L3 is not None
        assert earth_moon_system.L4 is not None
        assert earth_moon_system.L5 is not None

    def test_l4_l5_triangle_points(self, earth_moon_system):
        """Test L4 and L5 are equilateral triangle points"""
        earth_moon_system.compute_libration_points()

        # L4 should be at (0.5 - mu, sqrt(3)/2)
        expected_l4_y = np.sqrt(3) / 2
        assert abs(earth_moon_system.L4[1] - expected_l4_y) < 0.01

        # L5 should be at (0.5 - mu, -sqrt(3)/2)
        expected_l5_y = -np.sqrt(3) / 2
        assert abs(earth_moon_system.L5[1] - expected_l5_y) < 0.01

    def test_get_libration_point(self, earth_moon_system):
        """Test get_libration_point method"""
        L1 = earth_moon_system.get_libration_point(LibrationPoint.L1)
        assert L1 is not None
        assert len(L1) == 3

    def test_get_libration_point_auto_compute(self, earth_moon_system):
        """Test get_libration_point computes if not already computed"""
        # Don't call compute_libration_points first
        L1 = earth_moon_system.get_libration_point(LibrationPoint.L1)
        assert L1 is not None
        assert earth_moon_system.has_L_points is True

    def test_get_libration_point_invalid(self, earth_moon_system):
        """Test get_libration_point raises error for invalid point"""
        earth_moon_system.compute_libration_points()
        with pytest.raises(ValueError, match="无效的平动点"):
            earth_moon_system.get_libration_point("invalid")


class TestCR3BPSystemJacobiConstant:
    """Tests for get_jacobi_constant method"""

    def test_jacobi_constant_at_l1(self, earth_moon_system):
        """Test Jacobi constant at L1 point"""
        earth_moon_system.compute_libration_points()
        state = np.array([earth_moon_system.L1[0], 0, 0, 0, 0, 0])
        C = earth_moon_system.get_jacobi_constant(state)
        assert isinstance(C, float)
        assert C > 0  # L1 has positive Jacobi constant in normalized units

    def test_jacobi_constant_zero_velocity(self, earth_moon_system):
        """Test Jacobi constant with zero velocity"""
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        C = earth_moon_system.get_jacobi_constant(state)
        assert isinstance(C, float)

    def test_jacobi_constant_with_velocity(self, earth_moon_system):
        """Test Jacobi constant with non-zero velocity"""
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.1, 0.0])
        C = earth_moon_system.get_jacobi_constant(state)
        assert isinstance(C, float)

    def test_jacobi_constant_no_inf_at_body_position(self, earth_moon_system):
        """Jacobi constant at a body position should return nan with warning, not inf."""
        import warnings

        mu = earth_moon_system.mu
        state_at_primary = np.array([-mu, 0.0, 0.0, 0.0, 0.0, 0.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = earth_moon_system.get_jacobi_constant(state_at_primary)
            assert np.isnan(result), f"Expected NaN at singularity, got {result}"
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)


class TestCR3BPSystemCharacteristicScales:
    """Tests for set_characteristic_scales method"""

    def test_set_characteristic_scales(self, earth_moon_system):
        """Test setting characteristic scales"""
        distance = 384400  # km
        period = 27.32 * 86400  # seconds

        earth_moon_system.set_characteristic_scales(distance=distance, period=period)

        assert earth_moon_system.characteristic_length == distance
        assert earth_moon_system.characteristic_time is not None
        assert earth_moon_system.characteristic_velocity is not None
        assert earth_moon_system.is_initialized is True

    def test_characteristic_time_calculation(self, earth_moon_system):
        """Test characteristic time is period/(2*pi)"""
        distance = 384400
        period = 27.32 * 86400

        earth_moon_system.set_characteristic_scales(distance=distance, period=period)

        expected_time = period / (2 * np.pi)
        assert abs(earth_moon_system.characteristic_time - expected_time) < 1e-10

    def test_characteristic_velocity_calculation(self, earth_moon_system):
        """Test characteristic velocity is distance/time"""
        distance = 384400
        period = 27.32 * 86400

        earth_moon_system.set_characteristic_scales(distance=distance, period=period)

        expected_velocity = distance / earth_moon_system.characteristic_time
        assert abs(earth_moon_system.characteristic_velocity - expected_velocity) < 1e-10


class TestCR3BPSystemUnitConversion:
    """Tests for dimensionless/physical unit conversion"""

    def test_dimensionless_to_physical_requires_init(self, earth_moon_system):
        """Test conversion raises error without initialization"""
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="系统未初始化"):
            earth_moon_system.dimensionless_to_physical(state=state)

    def test_physical_to_dimensionless_requires_init(self, earth_moon_system):
        """Test conversion raises error without initialization"""
        state = np.array([192200, 0.0, 0.0, 0.0, 0.5, 0.0])  # km, km/s
        with pytest.raises(ValueError, match="系统未初始化"):
            earth_moon_system.physical_to_dimensionless(state=state)

    def test_dimensionless_to_physical_position(self, initialized_system):
        """Test dimensionless to physical position conversion"""
        dim_state = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        phys_state = initialized_system.dimensionless_to_physical(state=dim_state)

        expected_position = 0.5 * initialized_system.characteristic_length
        assert abs(phys_state[0] - expected_position) < 1e-10

    def test_physical_to_dimensionless_position(self, initialized_system):
        """Test physical to dimensionless position conversion"""
        phys_position = initialized_system.characteristic_length / 2
        phys_state = np.array([phys_position, 0.0, 0.0, 0.0, 0.0, 0.0])
        dim_state = initialized_system.physical_to_dimensionless(state=phys_state)

        assert abs(dim_state[0] - 0.5) < 1e-10

    def test_round_trip_conversion(self, initialized_system):
        """Test conversion round-trip is accurate"""
        original = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        converted = initialized_system.dimensionless_to_physical(state=original)
        back = initialized_system.physical_to_dimensionless(state=converted)

        assert np.allclose(original, back, atol=1e-10)


class TestCR3BPSystemPhysicalConstants:
    """Tests for class-level physical constants"""

    def test_gravitational_constant(self):
        """Test gravitational constant value"""
        assert CR3BP_System.G == 6.67430e-20

    def test_astronomical_unit(self):
        """Test astronomical unit value"""
        assert CR3BP_System.AU == 149597870.7

    def test_day_constant(self):
        """Test day in seconds"""
        assert CR3BP_System.DAY == 86400

    def test_year_constant(self):
        """Test year in seconds"""
        expected = 365.25 * 86400
        assert expected == CR3BP_System.YEAR
