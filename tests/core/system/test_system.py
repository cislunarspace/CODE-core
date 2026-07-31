"""CR3BP_System 单元测试。

覆盖平动点枚举、初始化、mu 校验、已知系统、
平动点计算、Jacobi 常数、特征尺度与单位转换。
"""

from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_System, LibrationPoint


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


class TestCR3BPMuValidation:
    """Tests for mu parameter validation"""

    def test_mu_validation_rejects_zero(self):
        """Test mu=0 is rejected"""
        with pytest.raises(ValueError, match="mu must be in"):
            CR3BP_System(mu=0.0, primary="A", secondary="B")

    def test_mu_validation_rejects_negative(self):
        """Test negative mu is rejected"""
        with pytest.raises(ValueError, match="mu must be in"):
            CR3BP_System(mu=-0.01, primary="A", secondary="B")

    def test_mu_validation_rejects_half(self):
        """Test mu=0.5 is rejected"""
        with pytest.raises(ValueError, match="mu must be in"):
            CR3BP_System(mu=0.5, primary="A", secondary="B")

    def test_mu_validation_rejects_out_of_range(self):
        """Test mu>0.5 is rejected"""
        with pytest.raises(ValueError, match="mu must be in"):
            CR3BP_System(mu=1.0, primary="A", secondary="B")

    def test_mu_validation_accepts_valid(self):
        """Test valid mu is accepted"""
        system = CR3BP_System(mu=0.012, primary="Earth", secondary="Moon")
        assert system.mu == 0.012


class TestCR3BPSystemKnownSystems:
    """Tests for CR3BP_System direct constructor + _with_default_scales helper."""

    def test_earth_moon_system(self, earth_moon_system):
        """Test Earth-Moon system creation"""
        assert earth_moon_system.mu == pytest.approx(1.21506683e-2, abs=1e-12)
        assert earth_moon_system.primary_body == "Earth"
        assert earth_moon_system.secondary_body == "Moon"

    def test_earth_moon_mu_full_precision(self):
        """Earth-Moon mu must match Cui et al. 2025 Table 1 to 10 digits."""
        system = CR3BP_System(
            mu=0.0121506683, primary="Earth", secondary="Moon"
        )._with_default_scales()
        assert system.mu == pytest.approx(1.21506683e-2, rel=0, abs=1e-12)

    def test_with_default_scales_auto_sets_scales(self):
        """_with_default_scales should auto-initialize characteristic scales."""
        system = CR3BP_System(
            mu=0.0121506683, primary="Earth", secondary="Moon"
        )._with_default_scales()
        assert system.is_initialized is True
        assert system.characteristic_length is not None
        assert system.characteristic_time is not None
        assert system.characteristic_velocity is not None

    def test_earth_moon_distance_matches_paper(self):
        """Earth-Moon characteristic length must be 384405 km (Cui et al. 2025)."""
        system = CR3BP_System(
            mu=0.0121506683, primary="Earth", secondary="Moon"
        )._with_default_scales()
        assert system.characteristic_length == pytest.approx(384405.0, abs=1e-6)

    def test_earth_moon_TU_days(self):
        """Earth-Moon characteristic time must match 4.34811305 days."""
        system = CR3BP_System(
            mu=0.0121506683, primary="Earth", secondary="Moon"
        )._with_default_scales()
        assert system.characteristic_time / 86400 == pytest.approx(4.34811305, abs=1e-8)

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
        """Test ValueError for unsupported body pair in _with_default_scales."""
        with pytest.raises(ValueError, match="_with_default_scales 不支持"):
            CR3BP_System(mu=0.01, primary="A", secondary="B")._with_default_scales()

    def test_known_systems_keys(self):
        """Test _with_default_scales supports expected primary/secondary pairs."""
        for primary, secondary in [("Earth", "Moon"), ("Sun", "Earth"), ("Sun", "Jupiter")]:
            mu = {
                ("Earth", "Moon"): 1.21506683e-2,
                ("Sun", "Earth"): 3.0039e-6,
                ("Sun", "Jupiter"): 0.0009535,
            }[(primary, secondary)]
            sys = CR3BP_System(mu=mu, primary=primary, secondary=secondary)._with_default_scales()
            assert sys.is_initialized is True


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

        expected_l4_y = np.sqrt(3) / 2
        assert abs(earth_moon_system.L4[1] - expected_l4_y) < 0.01

        expected_l5_y = -np.sqrt(3) / 2
        assert abs(earth_moon_system.L5[1] - expected_l5_y) < 0.01

    def test_get_libration_point(self, earth_moon_system):
        """Test get_libration_point method"""
        L1 = earth_moon_system.get_libration_point(LibrationPoint.L1)
        assert L1 is not None
        assert len(L1) == 3

    def test_get_libration_point_auto_compute(self, earth_moon_system):
        """Test get_libration_point computes if not already computed"""
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
        assert C > 0

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

    def test_set_characteristic_scales_rejects_non_positive(self):
        """Test that non-positive distance and period are rejected"""
        system = CR3BP_System(mu=0.012, primary="Earth", secondary="Moon")
        with pytest.raises(ValueError, match="distance must be positive"):
            system.set_characteristic_scales(distance=0, period=100)
        with pytest.raises(ValueError, match="period must be positive"):
            system.set_characteristic_scales(distance=100, period=-1)


class TestCR3BPSystemUnitConversion:
    """Tests for dimensionless/physical unit conversion"""

    def test_dimensionless_to_physical_requires_init(self):
        """Test conversion raises error without initialization"""
        system = CR3BP_System(mu=0.012, primary="Earth", secondary="Moon")
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="系统未初始化"):
            system.dimensionless_to_physical(state=state)

    def test_physical_to_dimensionless_requires_init(self):
        """Test conversion raises error without initialization"""
        system = CR3BP_System(mu=0.012, primary="Earth", secondary="Moon")
        state = np.array([192200, 0.0, 0.0, 0.0, 0.5, 0.0])  # km, km/s
        with pytest.raises(ValueError, match="系统未初始化"):
            system.physical_to_dimensionless(state=state)

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


class TestCR3BPSystemDUTUVUProperties:
    """Tests for DU/TU/VU property aliases (TOD convention: km, days, m/s)"""

    def test_DU_returns_kilometers(self, earth_moon_system):
        """DU returns characteristic length in km."""
        assert pytest.approx(384405.0, abs=1e-6) == earth_moon_system.DU

    def test_TU_returns_days(self, earth_moon_system):
        """TU returns characteristic time in days (TOD convention)."""
        assert pytest.approx(4.34811305, abs=1e-8) == earth_moon_system.TU

    def test_VU_returns_meters_per_second(self, earth_moon_system):
        """VU returns characteristic velocity in m/s (TOD convention)."""
        assert pytest.approx(1023.23281, abs=0.01) == earth_moon_system.VU


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


# --- Ported from tests/core/system/test_system_info.py ---
# info() method tests — unique coverage not in the main system test suite


@pytest.fixture
def uninitialized_system():
    """Create Earth-Moon system without initialized characteristic scales."""
    return CR3BP_System(mu=1.21506683e-2, primary="Earth", secondary="Moon")


@pytest.fixture
def full_system(initialized_system):
    """Create fully initialized system with libration points and mass info."""
    initialized_system.compute_libration_points()
    initialized_system.mass_primary = 5.972e24
    initialized_system.mass_secondary = 7.342e22
    initialized_system.total_mass = 5.972e24 + 7.342e22
    return initialized_system


def _capture_info(system, mode="default"):
    """Capture printed output from info() method."""
    buf = StringIO()
    with redirect_stdout(buf):
        system.info(mode=mode)
    return buf.getvalue()


class TestInfoDefault:
    """Tests for info() default mode"""

    def test_default_header_footer(self, earth_moon_system):
        """Default mode includes header and footer separators."""
        output = _capture_info(earth_moon_system)
        lines = output.strip().split("\n")
        assert lines[0] == "=" * 60
        assert lines[1] == "CR3BP 系统信息"
        assert lines[2] == "=" * 60
        assert lines[-1] == "=" * 60

    def test_default_basic_params(self, earth_moon_system):
        """Default mode includes basic parameters."""
        output = _capture_info(earth_moon_system)
        assert "Earth-Moon" in output
        assert "1.215067e-02" in output
        assert "主天体：Earth" in output
        assert "次天体：Moon" in output

    def test_default_no_extended_info(self, earth_moon_system):
        """Default mode does not include extended information."""
        output = _capture_info(earth_moon_system)
        assert "系统状态:" not in output
        assert "特征尺度:" not in output
        assert "平动点位置" not in output
        assert "质量信息:" not in output


class TestInfoAll:
    """Tests for info(mode='all') mode"""

    def test_all_includes_system_state(self, uninitialized_system):
        """All mode includes system state information."""
        output = _capture_info(uninitialized_system, mode="all")
        assert "系统状态:" in output
        assert "是否初始化：False" in output
        assert "是否已计算平动点：False" in output

    def test_all_uninitialized_scales(self, uninitialized_system):
        """All mode shows hint when characteristic scales are not set."""
        output = _capture_info(uninitialized_system, mode="all")
        assert "特征尺度：未设置" in output

    def test_all_initialized_scales(self, initialized_system):
        """All mode shows specific values when characteristic scales are set."""
        output = _capture_info(initialized_system, mode="all")
        assert "特征尺度:" in output
        assert "特征长度：384405.00 km" in output
        assert "特征速度" in output
        assert "平均角速度" in output
        assert "轨道周期" in output
        assert "半长轴：384405.00 km" in output

    def test_all_no_libration_points(self, uninitialized_system):
        """All mode shows hint when libration points are not computed."""
        output = _capture_info(uninitialized_system, mode="all")
        assert "平动点：未计算" in output

    def test_all_with_libration_points(self, full_system):
        """All mode shows coordinates when libration points are computed."""
        output = _capture_info(full_system, mode="all")
        assert "平动点位置 (无量纲坐标):" in output
        assert "L1:" in output
        assert "L2:" in output
        assert "L3:" in output
        assert "L4:" in output
        assert "L5:" in output

    def test_all_no_mass_info(self, earth_moon_system):
        """All mode shows hint when mass is not set."""
        output = _capture_info(earth_moon_system, mode="all")
        assert "质量信息：未设置" in output

    def test_all_with_mass_info(self, full_system):
        """All mode shows specific values when mass is set."""
        output = _capture_info(full_system, mode="all")
        assert "主天体质量" in output
        assert "次天体质量" in output
        assert "总质量" in output

    def test_all_state_flags_after_init(self, full_system):
        """All mode shows correct flags after full initialization."""
        output = _capture_info(full_system, mode="all")
        assert "是否初始化：True" in output
        assert "是否已计算平动点：True" in output


class TestInfoDifferentSystems:
    """Tests for info() output across different systems"""

    def test_sun_earth_system(self):
        system = CR3BP_System(
            mu=3.0039e-06, primary="Sun", secondary="Earth"
        )._with_default_scales()
        output = _capture_info(system)
        assert "Sun-Earth" in output

    def test_sun_jupiter_system(self):
        system = CR3BP_System(
            mu=0.0009535, primary="Sun", secondary="Jupiter"
        )._with_default_scales()
        output = _capture_info(system)
        assert "Sun-Jupiter" in output

    def test_custom_system(self):
        system = CR3BP_System(mu=0.001, primary="Star", secondary="Planet")
        output = _capture_info(system)
        assert "Star-Planet" in output
        assert "1.000000e-03" in output
