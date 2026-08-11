"""SolarRadiationPressure × ConicalShadowModel 纯集成测试。

验证阴影几何 ``_body_flux_factor`` 与多体合成 ``_combine_body_fluxes``
（纯 Python，保留）。SRP 加速度本身由 Rust 承载，不再通过 Python
``_compute_srp_acceleration`` 验证。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.forces.shadow import ConicalShadowModel
from e2m2e.data.constants import AU_KM as _AU_KM
from e2m2e.data.constants.bodies import EARTH, SUN

pytestmark = pytest.mark.force

_R_SUN = SUN.mean_radius_km
_R_EARTH = EARTH.gravity_ref_radius_km


def test_srp_stores_injected_shadow() -> None:
    """SRP 持有注入的阴影模型实例。"""
    from e2m2e.algorithm.forces.srp import SolarRadiationPressure

    shadow = ConicalShadowModel(bodies=["EARTH", "MOON"])
    srp = SolarRadiationPressure(area=10.0, mass=1000.0, shadow=shadow)
    assert srp.shadow is shadow


def test_shadow_umbra_gives_zero_flux() -> None:
    """本影几何：阴影 flux≈0。"""
    shadow = ConicalShadowModel()

    sun_pos = np.array([0.0, 0.0, 0.0])
    body_pos = np.array([_AU_KM, 0.0, 0.0])
    sc_pos = np.array([_AU_KM + 384000.0, 0.0, 0.0])  # 地球背日侧本影

    flux = shadow._body_flux_factor(sc_pos, body_pos, sun_pos, _R_EARTH, _R_SUN)
    assert flux == pytest.approx(0.0, abs=1e-12)


def test_shadow_full_sun_gives_full_flux() -> None:
    """全光照几何：阴影 flux=1。"""
    shadow = ConicalShadowModel()

    sun_pos = np.array([0.0, 0.0, 0.0])
    body_pos = np.array([_AU_KM, 0.0, 0.0])
    sc_pos = np.array([_AU_KM, 1.0e7, 0.0])  # 远离阴影锥

    flux = shadow._body_flux_factor(sc_pos, body_pos, sun_pos, _R_EARTH, _R_SUN)
    assert flux == pytest.approx(1.0, abs=1e-12)


def test_shadow_penumbra_gives_intermediate_flux() -> None:
    """半影几何：0 < flux < 1。"""
    shadow = ConicalShadowModel()

    sun_pos = np.array([0.0, 0.0, 0.0])
    body_pos = np.array([_AU_KM, 0.0, 0.0])
    sc_pos = np.array([_AU_KM + 1.0e6, 5000.0, 0.0])  # 半影

    flux = shadow._body_flux_factor(sc_pos, body_pos, sun_pos, _R_EARTH, _R_SUN)
    assert 0.0 < flux < 1.0
