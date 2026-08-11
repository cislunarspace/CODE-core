"""e2m2e/data/constants/ 单元测试。"""

import math

import pytest

from e2m2e.data.constants import (
    AU_KM,
    EARTH,
    EMB,
    GRAVITATIONAL_CONSTANT,
    KM_TO_M,
    MOON,
    SECONDS_PER_DAY,
    SECONDS_PER_JULIAN_YEAR,
    SOLAR_FLUX_TSI_W_M2,
    SOLAR_FLUX_W_M2,
    SOLAR_PRESSURE_1AU,
    SPEED_OF_LIGHT_KMS,
    SUN,
    Datum,
)

pytestmark = pytest.mark.data


class TestRustPythonConsistency:
    """Rust 与 Python 从同一 constants.toml 加载的常量必须逐位一致。"""

    @pytest.fixture(scope="class")
    def rust(self):
        from e2m2e._integrators import _propagation_constants

        return _propagation_constants.constant_value_py

    def test_universal_constants_match_rust(self, rust):
        assert rust("universal.speed_of_light_kms") == SPEED_OF_LIGHT_KMS
        assert rust("universal.gravitational_constant") == GRAVITATIONAL_CONSTANT
        assert rust("universal.au_km") == AU_KM
        assert rust("universal.seconds_per_day") == SECONDS_PER_DAY
        assert rust("universal.days_per_julian_year") == 365.25
        assert rust("universal.days_per_julian_century") == 36525.0
        assert rust("universal.km_to_m") == KM_TO_M
        assert rust("universal.solar_flux_w_m2") == SOLAR_FLUX_W_M2
        assert rust("universal.solar_flux_tsi_w_m2") == SOLAR_FLUX_TSI_W_M2
        assert rust("universal.solar_pressure_1au") == pytest.approx(SOLAR_PRESSURE_1AU)

    def test_datum_constants_match_rust(self, rust):
        assert rust("datum.DE421.mu") == Datum.DE421.mu
        assert rust("datum.DE421.earth_gm") == Datum.DE421.earth_gm
        assert rust("datum.DE421.moon_gm") == Datum.DE421.moon_gm
        assert rust("datum.DE421.sun_gm") == Datum.DE421.sun_gm
        assert rust("datum.DE421.emb_gm") == Datum.DE421.emb_gm
        assert rust("datum.DE421.char_length_km") == Datum.DE421.char_length_km
        assert rust("datum.DE421.char_time_s") == Datum.DE421.char_time_s

        assert rust("datum.DE440.mu") == Datum.DE440.mu
        assert rust("datum.DE440.earth_gm") == Datum.DE440.earth_gm
        assert rust("datum.DE440.moon_gm") == Datum.DE440.moon_gm
        assert rust("datum.DE440.sun_gm") == Datum.DE440.sun_gm
        assert rust("datum.DE440.emb_gm") == Datum.DE440.emb_gm

        assert rust("datum.WGS84.earth_gm") == Datum.WGS84.earth_gm
        assert rust("datum.WGS84.earth_radius_km") == Datum.WGS84.earth_radius_km
        assert rust("datum.WGS84.earth_flattening") == Datum.WGS84.earth_flattening

    def test_body_constants_match_rust(self, rust):
        assert rust("body.SUN.mean_radius_km") == SUN.mean_radius_km
        assert rust("body.SUN.naif_id") == SUN.naif_id

        assert rust("body.EARTH.mean_radius_km") == EARTH.mean_radius_km
        assert rust("body.EARTH.gravity_ref_radius_km") == EARTH.gravity_ref_radius_km
        assert rust("body.EARTH.flattening") == EARTH.flattening
        assert rust("body.EARTH.naif_id") == EARTH.naif_id
        assert rust("body.EARTH.rotation_rate_iers_rad_s") == EARTH.rotation_rate_iers_rad_s
        assert rust("body.EARTH.rotation_rate_gmat_rad_s") == EARTH.rotation_rate_gmat_rad_s
        assert rust("body.EARTH.gm.DE421") == EARTH.gm_by_datum["DE421"]
        assert rust("body.EARTH.gm.DE440") == EARTH.gm_by_datum["DE440"]
        assert rust("body.EARTH.gm.WGS84") == EARTH.gm_by_datum["WGS84"]

        assert rust("body.MOON.mean_radius_km") == MOON.mean_radius_km
        assert rust("body.MOON.gravity_ref_radius_km") == MOON.gravity_ref_radius_km
        assert rust("body.MOON.naif_id") == MOON.naif_id
        assert rust("body.MOON.gm.DE421") == MOON.gm_by_datum["DE421"]
        assert rust("body.MOON.gm.DE440") == MOON.gm_by_datum["DE440"]

        assert rust("body.EMB.naif_id") == EMB.naif_id
        assert rust("body.EMB.gm.DE421") == EMB.gm_by_datum["DE421"]
        assert rust("body.EMB.gm.DE440") == EMB.gm_by_datum["DE440"]


class TestUniversalConstants:
    def test_speed_of_light(self):
        assert SPEED_OF_LIGHT_KMS == 299792.458

    def test_gravitational_constant(self):
        assert GRAVITATIONAL_CONSTANT == 6.67430e-20

    def test_astronomical_unit(self):
        assert AU_KM == 149597870.7

    def test_time_constants(self):
        assert SECONDS_PER_DAY == 86400.0
        assert SECONDS_PER_JULIAN_YEAR == 365.25 * 86400.0

    def test_unit_conversion(self):
        assert KM_TO_M == 1000.0

    def test_solar_flux_and_pressure(self):
        assert SOLAR_FLUX_W_M2 == 1367.0
        assert SOLAR_FLUX_TSI_W_M2 == 1361.0
        expected_pressure = SOLAR_FLUX_W_M2 / (SPEED_OF_LIGHT_KMS * KM_TO_M)
        assert pytest.approx(SOLAR_PRESSURE_1AU) == expected_pressure


class TestDatumValues:
    def test_de421_values(self):
        assert Datum.DE421.mu == 0.012150585350562453
        assert Datum.DE421.earth_gm == 398600.4415
        assert Datum.DE421.moon_gm == 4902.8005821478
        assert Datum.DE421.sun_gm == 1.32712428e11
        assert Datum.DE421.emb_gm == 403503.242083
        assert Datum.DE421.char_length_km == 384400.0
        assert Datum.DE421.char_time_s == 375190.2588926273

    def test_de440_values(self):
        assert Datum.DE440.mu == 0.012150584394709708
        assert Datum.DE440.earth_gm == 398600.435507
        assert Datum.DE440.moon_gm == 4902.800118
        assert Datum.DE440.sun_gm == 1.32712440018e11
        assert Datum.DE440.emb_gm == 403503.235502

    def test_wgs84_values(self):
        assert Datum.WGS84.earth_gm == 398600.4418
        assert Datum.WGS84.earth_radius_km == 6378.137
        assert Datum.WGS84.earth_flattening == pytest.approx(1.0 / 298.257223563)


class TestDatumConsistency:
    def test_de421_mu_matches_gm_ratio(self):
        mu_computed = Datum.DE421.moon_gm / (Datum.DE421.earth_gm + Datum.DE421.moon_gm)
        assert Datum.DE421.mu == pytest.approx(mu_computed, rel=1e-15)

    def test_de421_char_time_matches_definition(self):
        total_gm = Datum.DE421.earth_gm + Datum.DE421.moon_gm
        char_time_computed = math.sqrt(Datum.DE421.char_length_km**3 / total_gm)
        assert Datum.DE421.char_time_s == pytest.approx(char_time_computed, rel=1e-12)

    def test_de440_mu_matches_gm_ratio(self):
        mu_computed = Datum.DE440.moon_gm / (Datum.DE440.earth_gm + Datum.DE440.moon_gm)
        assert Datum.DE440.mu == pytest.approx(mu_computed, rel=1e-15)


class TestBodies:
    def test_earth(self):
        assert EARTH.naif_id == 399
        assert EARTH.mean_radius_km == 6378.137
        assert EARTH.gravity_ref_radius_km == 6378.1363
        assert EARTH.rotation_rate_iers_rad_s == 7.292115146706979e-5
        assert EARTH.rotation_rate_gmat_rad_s == 7.29211585530e-5

    def test_moon(self):
        assert MOON.naif_id == 301
        assert MOON.mean_radius_km == 1737.4
        assert MOON.gravity_ref_radius_km == 1738.0

    def test_sun(self):
        assert SUN.naif_id == 10
        assert SUN.mean_radius_km == 696000.0

    def test_emb(self):
        assert EMB.naif_id == 3
        assert EMB.gm_by_datum["DE421"] == 403503.242083
        assert EMB.gm_by_datum["DE440"] == 403503.235502

    def test_planet_naif_ids(self):
        from e2m2e.data.constants import (
            JUPITER,
            MARS,
            MERCURY,
            NEPTUNE,
            PLUTO,
            SATURN,
            URANUS,
            VENUS,
        )

        assert MERCURY.naif_id == 199
        assert VENUS.naif_id == 299
        assert MARS.naif_id == 499
        assert JUPITER.naif_id == 599
        assert SATURN.naif_id == 699
        assert URANUS.naif_id == 799
        assert NEPTUNE.naif_id == 899
        assert PLUTO.naif_id == 999
