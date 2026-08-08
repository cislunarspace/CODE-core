"""ECOM 光压模型解析退化测试。

测试策略：解析退化验证——当 ECOM 的 DYB 系数全部退化（dyb[1..9]=0）时，
模型行为应与标准 cannonball SRP 完全一致（零容差）。

不使用 DFH 黄金样本。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.forces.ecom_srp import (
    EcomSolarRadiationPressure,
    _build_dyb_frame,
)
from e2m2e.algorithm.forces.srp import SolarRadiationPressure

pytestmark = [pytest.mark.l1]


class TestEcomConstruction:
    """构造与参数校验。"""

    def test_valid_construction(self):
        dyb = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ecom = EcomSolarRadiationPressure(dyb=dyb)
        assert ecom.dyb == dyb
        assert ecom.shadow is None

    def test_dyb_length_must_be_9(self):
        with pytest.raises(ValueError, match="9 elements"):
            EcomSolarRadiationPressure(dyb=[0.01] * 8)

    def test_dyb_returns_copy(self):
        dyb = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ecom = EcomSolarRadiationPressure(dyb=dyb)
        returned = ecom.dyb
        returned[0] = 999.0
        assert ecom.dyb[0] == 0.01


class TestDybFrame:
    """D-Y-B 坐标系构造。"""

    def test_orthogonality(self):
        """D, Y, B 三轴应互相正交。"""
        sc = np.array([42164.0, 0.0, 0.0])
        sun_to_sc = np.array([149597870.7, 0.0, 0.0])
        d, y, b = _build_dyb_frame(sc, sun_to_sc)
        assert abs(float(np.dot(d, y))) < 1e-12
        assert abs(float(np.dot(d, b))) < 1e-12
        assert abs(float(np.dot(y, b))) < 1e-12

    def test_unit_norms(self):
        """D, Y, B 应为单位向量。"""
        sc = np.array([42164.0, 0.0, 0.0])
        sun_to_sc = np.array([100000.0, 50000.0, -30000.0])
        d, y, b = _build_dyb_frame(sc, sun_to_sc)
        assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-12
        assert abs(float(np.linalg.norm(y)) - 1.0) < 1e-12
        assert abs(float(np.linalg.norm(b)) - 1.0) < 1e-12

    def test_d_hat_points_along_sun_to_sc(self):
        """D 应沿 Sun→SC 方向。"""
        sc = np.array([42164.0, 0.0, 0.0])
        sun_to_sc = np.array([1.0, 2.0, 3.0])
        d, _, _ = _build_dyb_frame(sc, sun_to_sc)
        expected = sun_to_sc / np.linalg.norm(sun_to_sc)
        np.testing.assert_allclose(d, expected, atol=1e-15)

    def test_degenerate_collinear(self):
        """D 与 SC 平行时仍能构造正交系。"""
        sc = np.array([1.0, 0.0, 0.0])
        sun_to_sc = np.array([100.0, 0.0, 0.0])
        d, y, b = _build_dyb_frame(sc, sun_to_sc)
        assert abs(float(np.dot(d, y))) < 1e-12
        assert abs(float(np.dot(d, b))) < 1e-12


class TestCannonballDegradation:
    """当 dyb[1..9]=0 时，ECOM 应退化为标准 cannonball SRP（零容差）。"""

    def _compare_accelerations(self, sc_pos, sun_to_sc, flux, cr, area, mass):
        """比较 ECOM 退化加速与 SRP 加速。"""
        # SRP cannonball
        srp = SolarRadiationPressure(area=area, mass=mass, cr=cr)
        a_srp = srp._compute_srp_acceleration(sun_to_sc, flux)

        # ECOM with dyb[0] = cr * area / mass（等效面质比），其余为零
        # DFH 约定：dyb[0] = 等效面质比，cr=1（已折入），mass=1
        # cannonball: a = flux * P * (AU/r)^2 * cr * area / mass / 1000 * û
        # ECOM: a = flux * P * (AU/r)^2 * dyb[0] / 1000 * (d_comp * d + ...)
        # 当 d_comp=1, y_comp=0, b_comp=0 时，
        # ECOM = flux * P * (AU/r)^2 * dyb[0] / 1000 * d_hat
        # cannonball = flux * P * (AU/r)^2 * cr * area / mass / 1000 * û
        # 所以 dyb[0] = cr * area / mass
        equivalent_a2m = cr * area / mass
        dyb = [equivalent_a2m] + [0.0] * 8
        ecom = EcomSolarRadiationPressure(dyb=dyb)
        a_ecom = ecom._compute_ecom_acceleration(sc_pos, sun_to_sc, flux)

        return a_srp, a_ecom

    def test_cannonball_degradation_zero_tolerance(self):
        """ECOM 退化 = cannonball SRP，零容差。"""
        sc_pos = np.array([42164.0, 0.0, 0.0])
        sun_to_sc = np.array([149597870.7, 0.0, 0.0])  # 1 AU
        flux = 1.0
        cr, area, mass = 1.5, 10.0, 1000.0

        a_srp, a_ecom = self._compare_accelerations(sc_pos, sun_to_sc, flux, cr, area, mass)
        np.testing.assert_array_equal(a_srp, a_ecom)

    def test_cannonball_degradation_with_shadow(self):
        """带阴影因子的退化一致性。"""
        sc_pos = np.array([42164.0, 0.0, 0.0])
        sun_to_sc = np.array([149597870.7, 0.0, 0.0])
        flux = 0.75  # 部分阴影
        cr, area, mass = 2.0, 5.0, 500.0

        a_srp, a_ecom = self._compare_accelerations(sc_pos, sun_to_sc, flux, cr, area, mass)
        np.testing.assert_array_equal(a_srp, a_ecom)

    def test_cannonball_degradation_oblique(self):
        """倾斜方向的退化一致性。"""
        sc_pos = np.array([30000.0, 20000.0, 10000.0])
        sun_to_sc = np.array([1e8, -5e7, 3e6])
        flux = 1.0
        cr, area, mass = 1.0, 20.0, 2000.0

        a_srp, a_ecom = self._compare_accelerations(sc_pos, sun_to_sc, flux, cr, area, mass)
        np.testing.assert_array_equal(a_srp, a_ecom)

    def test_cannonball_degradation_various_dyb0(self):
        """不同等效面质比的退化一致性。"""
        sc_pos = np.array([42164.0, 0.0, 0.0])
        sun_to_sc = np.array([1.496e8, 0.0, 0.0])
        flux = 1.0

        for a2m in [0.001, 0.01, 0.05, 0.1]:
            dyb = [a2m] + [0.0] * 8
            ecom = EcomSolarRadiationPressure(dyb=dyb)
            a_ecom = ecom._compute_ecom_acceleration(sc_pos, sun_to_sc, flux)

            # 等价 cannonball: cr=1, area=a2m, mass=1
            srp = SolarRadiationPressure(area=a2m, mass=1.0, cr=1.0)
            a_srp = srp._compute_srp_acceleration(sun_to_sc, flux)

            np.testing.assert_array_equal(a_srp, a_ecom)

    def test_zero_flux_yields_zero(self):
        """flux=0 时加速应为零。"""
        dyb = [0.01] + [0.0] * 8
        ecom = EcomSolarRadiationPressure(dyb=dyb)
        a = ecom._compute_ecom_acceleration([42164.0, 0.0, 0.0], [1e8, 0.0, 0.0], 0.0)
        np.testing.assert_array_equal(a, np.zeros(3))


class TestEcomPeriodicTerms:
    """周期项分量影响测试。"""

    def test_d_periodic_terms_affect_output(self):
        """D 方向周期项非零时，输出应不同于 cannonball。"""
        sc_pos = np.array([42164.0, 0.0, 0.0])
        sun_to_sc = np.array([1e8, 0.0, 0.0])
        flux = 1.0

        dyb_cannon = [0.01] + [0.0] * 8
        ecom_cannon = EcomSolarRadiationPressure(dyb=dyb_cannon)
        a_cannon = ecom_cannon._compute_ecom_acceleration(sc_pos, sun_to_sc, flux)

        # u=0 时 cos(u)=1, sin(u)=0，所以 dyb[1] 有贡献，dyb[2] 无
        dyb_d = [0.01, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ecom_d = EcomSolarRadiationPressure(dyb=dyb_d)
        a_d = ecom_d._compute_ecom_acceleration(sc_pos, sun_to_sc, flux)

        # D 方向系数非零应改变加速度（小值用 atol=0 严格比较）
        assert not np.allclose(a_cannon, a_d, atol=0.0)

    def test_y_component_affects_output(self):
        """Y 方向分量非零时，输出应改变。"""
        sc_pos = np.array([42164.0, 0.0, 0.0])
        sun_to_sc = np.array([1e8, 0.0, 0.0])
        flux = 1.0

        # u=0 → cos(u)=1，dyb[5] 有贡献
        dyb_y = [0.01, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
        ecom_cannon = EcomSolarRadiationPressure(dyb=[0.01] + [0.0] * 8)
        ecom_y = EcomSolarRadiationPressure(dyb=dyb_y)

        a_cannon = ecom_cannon._compute_ecom_acceleration(sc_pos, sun_to_sc, flux)
        a_y = ecom_y._compute_ecom_acceleration(sc_pos, sun_to_sc, flux)
        assert not np.allclose(a_cannon, a_y, atol=0.0)

    def test_b_component_affects_output(self):
        """B 方向分量非零时，输出应改变。"""
        sc_pos = np.array([42164.0, 0.0, 0.0])
        sun_to_sc = np.array([1e8, 0.0, 0.0])
        flux = 1.0

        # dyb[7] = 常量 B 方向
        dyb_b = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]
        ecom_cannon = EcomSolarRadiationPressure(dyb=[0.01] + [0.0] * 8)
        ecom_b = EcomSolarRadiationPressure(dyb=dyb_b)

        a_cannon = ecom_cannon._compute_ecom_acceleration(sc_pos, sun_to_sc, flux)
        a_b = ecom_b._compute_ecom_acceleration(sc_pos, sun_to_sc, flux)
        assert not np.allclose(a_cannon, a_b, atol=0.0)


class TestEcomToRustSpec:
    """to_rust_spec 序列化。"""

    def test_to_rust_spec_format(self):
        dyb = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ecom = EcomSolarRadiationPressure(dyb=dyb)
        spec = ecom.to_rust_spec()
        assert spec[0] == "ecom_srp"
        assert spec[1] == dyb
        assert spec[2] == []

    def test_to_rust_spec_with_shadow(self):
        from e2m2e.algorithm.forces.shadow import ConicalShadowModel

        shadow = ConicalShadowModel(bodies=["EARTH"])
        dyb = [0.01] + [0.0] * 8
        ecom = EcomSolarRadiationPressure(dyb=dyb, shadow=shadow)
        spec = ecom.to_rust_spec()
        assert spec[0] == "ecom_srp"
        assert spec[1] == dyb
        assert spec[2] == ["EARTH"]


class TestEcomConfigRoundTrip:
    """配置 round-trip 序列化（零容差）。"""

    def test_round_trip_no_shadow(self):
        """无阴影模型的 round-trip。"""
        dyb = [0.02, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
        ecom = EcomSolarRadiationPressure(dyb=dyb)
        config = ecom.to_config()
        ecom2 = EcomSolarRadiationPressure.from_config(config)
        assert ecom2.dyb == ecom.dyb
        assert ecom2.shadow is None

    def test_round_trip_dict_equal(self):
        """to_config(from_config(config)) == config（零容差）。"""
        dyb = [0.02, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
        ecom = EcomSolarRadiationPressure(dyb=dyb)
        config = ecom.to_config()
        ecom2 = EcomSolarRadiationPressure.from_config(config)
        config2 = ecom2.to_config()
        assert config == config2

    def test_round_trip_with_shadow(self):
        """带阴影模型的 round-trip。"""
        from e2m2e.algorithm.forces.shadow import ConicalShadowModel

        shadow = ConicalShadowModel(bodies=["EARTH", "MOON"])
        dyb = [0.01] + [0.0] * 8
        ecom = EcomSolarRadiationPressure(dyb=dyb, shadow=shadow)
        config = ecom.to_config()
        ecom2 = EcomSolarRadiationPressure.from_config(config)
        assert ecom2.dyb == ecom.dyb
        assert ecom2.shadow is not None
        assert list(ecom2.shadow.bodies) == ["EARTH", "MOON"]
        config2 = ecom2.to_config()
        assert config == config2

    def test_round_trip_zero_tolerance(self):
        """数值零容差验证。"""
        dyb = [0.0123456789, 1e-6, -2e-6, 3e-6, -4e-6, 5e-6, -6e-6, 7e-6, -8e-6]
        ecom = EcomSolarRadiationPressure(dyb=dyb)
        config = ecom.to_config()
        ecom2 = EcomSolarRadiationPressure.from_config(config)
        for i in range(9):
            assert ecom2.dyb[i] == pytest.approx(ecom.dyb[i], abs=0.0)


class TestEcomForceConfigIntegration:
    """与 force_config 序列化系统的集成测试。"""

    def test_serialize_deserialize(self):
        from e2m2e.algorithm.forces.force_config import build_force, serialize_force

        dyb = [0.02, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
        ecom = EcomSolarRadiationPressure(dyb=dyb)
        serialized = serialize_force(ecom)
        assert serialized["type"] == "EcomSolarRadiationPressure"
        assert serialized["params"]["dyb"] == dyb
        assert serialized["params"]["shadow"] is None

        rebuilt = build_force("EcomSolarRadiationPressure", serialized["params"])
        assert isinstance(rebuilt, EcomSolarRadiationPressure)
        assert rebuilt.dyb == dyb

    def test_force_model_round_trip(self):
        """ForceModel 级别的 round-trip。"""
        from e2m2e.algorithm.forces import ForceModel

        dyb = [0.01] + [0.0] * 8
        ecom = EcomSolarRadiationPressure(dyb=dyb)

        class _FakeSystem:
            coordinate_system = object()

        system = _FakeSystem()
        fm = ForceModel(system, [ecom])
        config = fm.to_config()
        fm2 = ForceModel.from_config(config, system)
        assert fm2.to_config() == config
