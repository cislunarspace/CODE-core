"""SolarRadiationPressure × ConicalShadowModel 纯集成测试。

验证阴影输出正确调制 SRP 加速度。
"""

from __future__ import annotations

import numpy as np

from e2m2e.algorithm.forces.shadow import ConicalShadowModel
from e2m2e.algorithm.forces.srp import SolarRadiationPressure
from e2m2e.data.templates.systems import AU as _AU_KM
from e2m2e.data.templates.systems import R_EARTH as _R_EARTH

_R_SUN = 695700.0
_P_SRP_1AU = 4.56e-6


def test_srp_stores_injected_shadow() -> None:
    """SRP 持有注入的阴影模型实例。"""
    shadow = ConicalShadowModel(bodies=["EARTH", "MOON"])
    srp = SolarRadiationPressure(area=10.0, mass=1000.0, shadow=shadow)
    assert srp.shadow is shadow


def test_srp_shadow_umbra_gives_zero_acceleration() -> None:
    """本影几何：阴影 flux≈0 → 合成加速度 ≈ 0。"""
    shadow = ConicalShadowModel()
    srp = SolarRadiationPressure(area=10.0, mass=1000.0, cr=1.5, shadow=shadow)

    sun_pos = np.array([0.0, 0.0, 0.0])
    body_pos = np.array([_AU_KM, 0.0, 0.0])
    sc_pos = np.array([_AU_KM + 384000.0, 0.0, 0.0])  # 地球背日侧本影

    sun_to_sc = sc_pos - sun_pos
    flux = shadow._body_flux_factor(sc_pos, body_pos, sun_pos, _R_EARTH, _R_SUN)
    acc = srp._compute_srp_acceleration(sun_to_sc, flux)

    np.testing.assert_array_equal(acc, np.zeros(3))


def test_srp_shadow_full_sun_gives_full_acceleration() -> None:
    """全光照几何：阴影 flux=1 → 合成加速度 = 满 SRP（P·Cr·A/m·(AU/r)²）。"""
    cr, area, mass = 1.5, 10.0, 1000.0
    shadow = ConicalShadowModel()
    srp = SolarRadiationPressure(area=area, mass=mass, cr=cr, shadow=shadow)

    sun_pos = np.array([0.0, 0.0, 0.0])
    body_pos = np.array([_AU_KM, 0.0, 0.0])
    sc_pos = np.array([_AU_KM, 1.0e7, 0.0])  # 远离阴影锥

    sun_to_sc = sc_pos - sun_pos
    r = np.linalg.norm(sun_to_sc)
    flux = shadow._body_flux_factor(sc_pos, body_pos, sun_pos, _R_EARTH, _R_SUN)
    assert flux == 1.0  # 确认全光照前提

    acc = srp._compute_srp_acceleration(sun_to_sc, flux)
    expected_si = _P_SRP_1AU * (_AU_KM / r) ** 2 * cr * area / mass
    np.testing.assert_allclose(np.linalg.norm(acc), expected_si / 1000.0, rtol=1e-12)


def test_srp_shadow_penumbra_gives_intermediate_acceleration() -> None:
    """半影几何：0 < flux < 1 → 合成加速度介于 0 与满 SRP 之间。"""
    shadow = ConicalShadowModel()
    srp = SolarRadiationPressure(area=10.0, mass=1000.0, cr=1.5, shadow=shadow)

    sun_pos = np.array([0.0, 0.0, 0.0])
    body_pos = np.array([_AU_KM, 0.0, 0.0])
    sc_pos = np.array([_AU_KM + 1.0e6, 5000.0, 0.0])  # 半影

    sun_to_sc = sc_pos - sun_pos
    flux = shadow._body_flux_factor(sc_pos, body_pos, sun_pos, _R_EARTH, _R_SUN)
    assert 0.0 < flux < 1.0

    acc_dark = np.linalg.norm(srp._compute_srp_acceleration(sun_to_sc, 0.0))
    acc_full = np.linalg.norm(srp._compute_srp_acceleration(sun_to_sc, 1.0))
    acc = np.linalg.norm(srp._compute_srp_acceleration(sun_to_sc, flux))

    assert acc_dark < acc < acc_full
