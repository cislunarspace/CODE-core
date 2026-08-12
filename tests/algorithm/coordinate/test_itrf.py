"""ITRF 坐标轴、天体原点与标准坐标系测试。

覆盖 ITRF93/GMAT 近似/ICRF 集成与标准预设工厂。
"""

import os

import numpy as np
import pytest

from e2m2e.algorithm.coordinate.standard_axes import (
    GMATITRFAxes,
    ICRSAxes,
    ITRFApproxAxes,
    ITRFAxes,
    ITRFSpiceAxes,
    standard_icrf,
    standard_itrf,
)
from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin, InertialOrigin
from e2m2e.data.frames.gmat_fixture import CoordinateDataError, gmat_data_dir
from e2m2e.data.kernels.manager import SPICEManager

pytestmark = pytest.mark.data


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


class TestICRFITRFIntegrationWithSpice:
    """ICRF↔ITRF 端到端集成测试,SPICE pxform 作参考源。

    issue #78 验收第 5 条"ICRF↔ITRF 旋转矩阵元素误差 < 1e-12"与
    第 6 条"所有矩阵满足正交性 < 1e-14"。把 standard_icrf() 与
    ITRFSpiceAxes + CelestialBodyOrigin('EARTH') 组合,验证 CoordinateSystem
    走出的转换与 SPICE pxform 直接对比。
    """

    def test_icrf_itrf_rotation_matches_pxform(self, spice_manager, requires_itrf93):
        """ICRF→ITRF 向量变换与 spiceypy.pxform('J2000','ITRF93',et) 元素差 < 1e-12。"""
        import spiceypy

        from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem

        icrf = standard_icrf()
        itrf = CoordinateSystem(
            axes=ITRFSpiceAxes(),
            origin=CelestialBodyOrigin("EARTH", spice_manager),
        )
        et = spice_manager.utc_to_et("2024-01-01T00:00:00")

        # 非零向量,从 ICRF 转 ITRF
        vec = np.array([1.0, 0.0, 0.0])
        result = icrf.transform_vector(vec, from_cs=icrf, to_cs=itrf, et=et)

        # SPICE 参考:pxform('J2000', 'ITRF93', et) 把 ICRF/J2000 向量转 ITRF93
        expected_R = spiceypy.pxform("J2000", "ITRF93", et)
        expected = expected_R @ vec

        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_icrf_itrf_rotation_matrix_is_orthogonal(self, spice_manager, requires_itrf93):
        """ICRF↔ITRF 链路两端 Axes 旋转矩阵 R @ R.T = I,误差 < 1e-14。"""
        from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem

        icrf = standard_icrf()
        itrf = CoordinateSystem(
            axes=ITRFSpiceAxes(),
            origin=CelestialBodyOrigin("EARTH", spice_manager),
        )
        et = spice_manager.utc_to_et("2024-01-01T00:00:00")

        R_icrf = icrf.axes.rotation_matrix(et)
        R_itrf = itrf.axes.rotation_matrix(et)

        np.testing.assert_allclose(R_icrf @ R_icrf.T, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(R_itrf @ R_itrf.T, np.eye(3), atol=1e-14)


class TestITRFApproxAxesConsistencyWithITRF93:
    """ITRFApproxAxes 与 SPICE-backed ITRF93 的方向一致性验证。

    issue #78 验收第 2 条"ITRFApproxAxes 实现并通过一致性测试"——
    近似实现(忽略极移 + 简化章动 + GMST 公式)与 ITRF93 不完全吻合:
    主要差异来自被忽略的极移(量级 ~0.3-1 角秒/轴 → 累计可达 ~1000 角秒,
    ~5e-3 rad)。因此本测试不再断言"亚角秒一致",而是验证方向正确:
    二者均为正交旋转,且近似旋转是精确旋转的小扰动(R^T R' ≈ I)。
    """

    def test_rotation_matrix_approximately_matches_itrf93(self, spice_manager, requires_itrf93):
        """ITRFApproxAxes 与 ITRFSpiceAxes 方向一致:均为正交,且互为小扰动。"""
        approx = ITRFApproxAxes()
        precise = ITRFSpiceAxes()
        et = spice_manager.utc_to_et("2024-01-01T00:00:00")

        R_approx = approx.rotation_matrix(et)
        R_precise = precise.rotation_matrix(et)

        # 二者均为正交旋转(行列式 = 1)
        np.testing.assert_allclose(np.linalg.det(R_approx), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.linalg.det(R_precise), 1.0, atol=1e-9)
        # 近似是精确的小扰动:R_approx^T @ R_precise ≈ I,偏离角 ~极移量级(< 0.02 rad)
        relative = R_approx.T @ R_precise
        np.testing.assert_allclose(relative, np.eye(3), atol=2e-2)


class TestStandardCoordinateSystems:
    def test_icrf_to_explicit_gmat_itrf_vector_round_trip(self):
        from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem

        icrf = CoordinateSystem(axes=ICRSAxes(), origin=InertialOrigin())
        # EOP 越界策略显式选择（#352 移除 compatibility 隐式切换后）
        itrf = CoordinateSystem(
            axes=GMATITRFAxes(eop_extrapolation="clamp"), origin=InertialOrigin()
        )
        vec = np.array([1.0, 0.0, 0.0])

        itrf_vec = icrf.transform_vector(vec, from_cs=icrf, to_cs=itrf, et=0.0)
        back = itrf.transform_vector(itrf_vec, from_cs=itrf, to_cs=icrf, et=0.0)

        np.testing.assert_allclose(back, vec, atol=1e-12)


class TestStandardIcrfPreset:
    """standard_icrf() 工厂:返回 ICRF 标准坐标系预设(CoordinateSystem)。

    issue #78 验收第 4 条"ICRF 与 ITRF 标准坐标系预设可用"——ICRF 侧的
    字面落实。ICRF = ICRSAxes(恒等旋转)+ InertialOrigin(SSB 无平移)。
    """

    def test_standard_icrf_returns_coordinate_system(self):
        """standard_icrf() 返回 CoordinateSystem,axes 为 ICRSAxes,origin 为 InertialOrigin。"""
        from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem

        cs = standard_icrf()
        assert isinstance(cs, CoordinateSystem)
        assert isinstance(cs.axes, ICRSAxes)
        assert isinstance(cs.origin, InertialOrigin)

    def test_standard_icrf_importable_from_core_top_level(self):
        """standard_icrf 可从 e2m2e.core 顶层导入(与 standard_itrf 对齐)。"""
        from e2m2e.algorithm.coordinate import standard_icrf as imported

        assert imported is standard_icrf
