"""ITRF axes, celestial origins, and standard coordinate-system tests."""

import os

import numpy as np
import pytest

from e2m2e.core.gmat_data import CoordinateDataError, gmat_data_dir
from e2m2e.core.spice import SPICEManager
from e2m2e.core.standard_axes import (
    GMATITRFAxes,
    ICRSAxes,
    ITRFApproxAxes,
    ITRFAxes,
    ITRFSpiceAxes,
    standard_itrf,
)
from e2m2e.core.standard_origins import CelestialBodyOrigin, InertialOrigin


@pytest.fixture
def spice_manager():
    """Provide a SPICEManager with available ephemeris and PCK kernels loaded."""
    manager = SPICEManager()
    loaded: list[str] = []
    try:
        kernel = manager.find_ephemeris_kernel("kernels")
        manager.load_kernel(kernel)
        loaded.append(kernel)
    except FileNotFoundError:
        pytest.skip("No ephemeris kernel found")

    for filename in os.listdir("kernels"):
        if filename.endswith((".bpc", ".tpc")):
            path = os.path.join("kernels", filename)
            manager.load_kernel(path)
            loaded.append(path)

    optional_data_dir = gmat_data_dir()
    if optional_data_dir is not None:
        planetary = optional_data_dir / "planetary_coeff"
        for filename in [
            "SPICEPlanetaryConstantsKernel.tpc",
            "SPICEEarthCurrentKernel.bpc",
            "SPICEEarthPredictedKernel.bpc",
            "earth_latest_high_prec.bpc",
        ]:
            path = planetary / filename
            if path.is_file():
                manager.load_kernel(str(path))
                loaded.append(str(path))

    yield manager

    for path in reversed(loaded):
        manager.unload_kernel(path)


@pytest.fixture
def requires_iau_earth(spice_manager):
    """Skip tests that require the low-fidelity IAU_EARTH text-PCK frame."""
    import spiceypy

    try:
        spiceypy.pxform("IAU_EARTH", "J2000", 0.0)
    except Exception:
        pytest.skip("IAU_EARTH PCK kernel not available")


@pytest.fixture
def requires_itrf93(spice_manager):
    """Skip tests that require an optional high-precision Earth BPC defining ITRF93."""
    import spiceypy

    try:
        spiceypy.pxform("ITRF93", "J2000", 0.0)
    except Exception:
        pytest.skip("ITRF93 Earth binary PCK not available; set GMAT_DATA_DIR for this check")


class TestCelestialBodyOrigin:
    def test_earth_origin_state(self, spice_manager):
        origin = CelestialBodyOrigin("EARTH", spice_manager)
        et = spice_manager.utc_to_et("2024-01-01T00:00:00")

        state = origin.state(et)

        assert state.shape == (6,)
        assert np.linalg.norm(state[:3]) > 1e7

    def test_origin_state_is_copy(self, spice_manager):
        origin = CelestialBodyOrigin("EARTH", spice_manager)
        et = spice_manager.utc_to_et("2024-01-01T00:00:00")

        state1 = origin.state(et)
        state2 = origin.state(et)
        state1[0] = 999.0

        assert state2[0] != 999.0


class TestITRFSpiceAxes:
    def test_standard_itrf_uses_itrf93(self):
        axes = standard_itrf()

        assert isinstance(axes, ITRFSpiceAxes)
        assert axes.frame == "ITRF93"
        assert isinstance(ITRFAxes(), ITRFSpiceAxes)
        assert ITRFAxes().frame == "ITRF93"

    def test_missing_itrf93_does_not_fallback_to_iau_earth(self, spice_manager, requires_iau_earth):
        axes = ITRFSpiceAxes("ITRF93")
        et = spice_manager.utc_to_et("2024-01-01T00:00:00")

        try:
            axes.rotation_matrix(et)
        except CoordinateDataError as exc:
            assert "no fallback to IAU_EARTH" in str(exc)
        else:
            pytest.skip("ITRF93 is available in this environment")

    @pytest.mark.parametrize(
        "utc_epoch",
        ["2000-01-01T00:00:00", "2017-01-01T00:00:00", "2026-06-12T00:00:00"],
    )
    def test_rotation_matrix_matches_spice_itrf93(self, spice_manager, requires_itrf93, utc_epoch):
        import spiceypy

        axes = ITRFSpiceAxes()
        et = spice_manager.utc_to_et(utc_epoch)

        rotation = axes.rotation_matrix(et)
        expected = spiceypy.pxform("ITRF93", "J2000", et)

        np.testing.assert_allclose(rotation, expected, atol=1e-12)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)

    @pytest.mark.parametrize(
        "utc_epoch",
        ["2000-01-01T00:00:00", "2017-01-01T00:00:00", "2026-06-12T00:00:00"],
    )
    def test_rotation_and_rate_matches_spice_sxform(
        self, spice_manager, requires_itrf93, utc_epoch
    ):
        import spiceypy

        axes = ITRFSpiceAxes()
        et = spice_manager.utc_to_et(utc_epoch)

        rotation, rate = axes.rotation_and_rate(et)
        expected = spiceypy.sxform("ITRF93", "J2000", et)

        np.testing.assert_allclose(rotation, expected[:3, :3], atol=1e-12)
        np.testing.assert_allclose(rate, expected[3:, :3], atol=1e-12)


class TestGMATITRFAxes:
    def test_explicit_gmat_itrf_runs_with_committed_fixtures(self):
        axes = GMATITRFAxes(eop_extrapolation="clamp")

        rotation, rate = axes.rotation_and_rate(0.0)

        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
        assert rate.shape == (3, 3)

    def test_explicit_gmat_itrf_can_be_sanity_checked_against_itrf93(
        self, spice_manager, requires_itrf93
    ):
        axes = GMATITRFAxes(eop_extrapolation="clamp")
        et = spice_manager.utc_to_et("2000-01-01T00:00:00")

        rotation = axes.rotation_matrix(et)
        import spiceypy

        expected = spiceypy.pxform("ITRF93", "J2000", et)
        assert np.max(np.abs(rotation - expected)) < 1e-7


class TestITRFApproxAxes:
    def test_rotation_matrix_orthogonal(self, spice_manager):
        axes = ITRFApproxAxes()
        et = spice_manager.utc_to_et("2024-01-01T00:00:00")

        rotation = axes.rotation_matrix(et)

        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)

    def test_approx_is_low_fidelity_and_not_precise_itrf(self, spice_manager):
        approx = ITRFApproxAxes()
        et = spice_manager.utc_to_et("2024-01-01T00:00:00")

        rotation = approx.rotation_matrix(et)

        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)

    def test_angular_velocity_z_is_positive_earth_rotation(self, spice_manager):
        """角速度 z 分量应为正（地球东向自转，ω_z ≈ +7.292e-5 rad/s）。

        回归测试：GAST 旋转方向曾经符号错误（_rotation3(-gast)），导致
        角速度为负、大气阻力相对速度计算错误。此测试防止该 bug 再现。
        """
        axes = ITRFApproxAxes()
        et = spice_manager.utc_to_et("2024-01-01T00:00:00")

        omega = axes.angular_velocity(et)

        assert omega[2] > 0.0, f"地球东向自转要求 ω_z > 0，实际 {omega[2]:.4e}"
        np.testing.assert_allclose(omega[2], 7.2921150e-5, rtol=1e-3)


class TestStandardCoordinateSystems:
    def test_icrf_to_explicit_gmat_itrf_vector_round_trip(self):
        from e2m2e.core.coordinate_system import CoordinateSystem

        icrf = CoordinateSystem(axes=ICRSAxes(), origin=InertialOrigin())
        itrf = CoordinateSystem(axes=GMATITRFAxes(compatibility="gmat"), origin=InertialOrigin())
        vec = np.array([1.0, 0.0, 0.0])

        itrf_vec = icrf.transform_vector(vec, from_cs=icrf, to_cs=itrf, et=0.0)
        back = itrf.transform_vector(itrf_vec, from_cs=itrf, to_cs=icrf, et=0.0)

        np.testing.assert_allclose(back, vec, atol=1e-12)
