"""SRP / Shadow 系统感知路径集成测试（需 SPICE 内核）。

验证 Rust ``srp_acceleration`` 绑定通过 SPICE 取太阳位置后，与阴影几何
``flux_factor`` 及 cannonball 公式一致。
"""

from __future__ import annotations

import numpy as np
import pytest
from e2m2e._integrators import srp_acceleration

from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
from e2m2e.algorithm.dynamics.ephemeris_system import EphemerisSystem
from e2m2e.algorithm.forces.shadow import ConicalShadowModel
from e2m2e.data.kernels.manager import SPICEManager

pytestmark = pytest.mark.force


_R_EARTH = 6378.1363
_R_SUN = 695700.0
_P_SRP_1AU = 4.56e-6
_AU_KM = 149597870.691


@pytest.fixture
def earth_icrf_system(spice_kernel_path):
    """地球中心 ICRF 传播系统（地月日三星历）。"""
    spice = SPICEManager()
    spice.load_kernel(spice_kernel_path)
    try:
        system = EphemerisSystem(
            bodies=["EARTH", "MOON", "SUN"],
            spice=spice,
            origin="EARTH",
        )
        system.coordinate_system = CoordinateSystem(
            axes=ICRSAxes(),
            origin=CelestialBodyOrigin(body="EARTH", spice=spice),
        )
        yield system
    finally:
        spice.unload_kernel(spice_kernel_path)


def _sun_pos_rel_earth(system, et):
    """太阳相对地心的 J2000 位置（km）。"""
    return system.spice.get_body_state("SUN", et, "J2000", "EARTH")[:3]


@pytest.mark.spice
def test_shadow_flux_factor_matches_manual(earth_icrf_system) -> None:
    """flux_factor(t, state, system) 与手算（SPICE 取位 + _body_flux_factor）一致。"""
    system = earth_icrf_system
    et = system.spice.utc_to_et("2025-06-21T11:00:06")

    # 400 km 圆轨道位置（任意相位）
    sc_pos = np.array([6778.0, 0.0, 0.0])
    state = np.array([6778.0, 0.0, 0.0, 0.0, 7.7, 0.0])

    shadow = ConicalShadowModel(bodies=["EARTH"])
    flux_system = shadow.flux_factor(et, state, system)

    sun_pos = _sun_pos_rel_earth(system, et)
    flux_manual = shadow._body_flux_factor(sc_pos, np.zeros(3), sun_pos, _R_EARTH, _R_SUN)
    np.testing.assert_allclose(flux_system, flux_manual, rtol=1e-12)


@pytest.mark.spice
def test_shadow_flux_factor_umbra_for_anti_sun_leo(earth_icrf_system) -> None:
    """反日点 LEO（400 km）→ 系统路径 flux_factor ≈ 0（地影）。"""
    system = earth_icrf_system
    et = system.spice.utc_to_et("2025-06-21T11:00:06")

    sun_pos = _sun_pos_rel_earth(system, et)
    sun_dir = sun_pos / np.linalg.norm(sun_pos)
    # 反日方向、400 km 高度
    sc_pos = -sun_dir * 6778.0
    state = np.array([*sc_pos, 0.0, 0.0, 0.0])

    shadow = ConicalShadowModel(bodies=["EARTH"])
    flux = shadow.flux_factor(et, state, system)
    assert flux == pytest.approx(0.0, abs=1e-9)


@pytest.mark.spice
def test_srp_rust_binding_matches_cannonball_formula(earth_icrf_system) -> None:
    """Rust ``srp_acceleration`` 与 cannonball 公式 + 阴影 flux 一致。"""
    system = earth_icrf_system
    et = system.spice.utc_to_et("2025-06-21T11:00:06")
    state = np.array([6778.0, 0.0, 0.0, 0.0, 7.7, 0.0])

    area, mass, cr = 10.0, 1000.0, 1.5
    acc = srp_acceleration(et, state[:3].tolist(), area, mass, cr, ["EARTH"], "EARTH")

    # 手算
    sun_pos = _sun_pos_rel_earth(system, et)
    sun_to_sc = state[:3] - sun_pos
    r = np.linalg.norm(sun_to_sc)
    shadow = ConicalShadowModel(bodies=["EARTH"])
    flux = shadow.flux_factor(et, state, system)
    expected_si = flux * _P_SRP_1AU * (_AU_KM / r) ** 2 * cr * area / mass
    expected_km = expected_si / 1000.0
    expected_dir = sun_to_sc / r

    np.testing.assert_allclose(np.linalg.norm(acc), expected_km, rtol=1e-10)
    cos_angle = np.dot(acc, expected_dir) / (np.linalg.norm(acc) * np.linalg.norm(expected_dir))
    assert cos_angle == pytest.approx(1.0, abs=1e-9)


@pytest.mark.spice
def test_srp_acceleration_points_away_from_sun(earth_icrf_system) -> None:
    """SRP 加速度在 ICRF 中指向远离太阳（与 Sun→SC 同向）。"""
    system = earth_icrf_system
    et = system.spice.utc_to_et("2025-06-21T11:00:06")
    state = np.array([6778.0, 0.0, 0.0, 0.0, 7.7, 0.0])

    acc = srp_acceleration(et, state[:3].tolist(), 10.0, 1000.0, 1.5, [], "EARTH")

    sun_pos = _sun_pos_rel_earth(system, et)
    sun_to_sc = state[:3] - sun_pos
    cos_angle = np.dot(acc, sun_to_sc) / (np.linalg.norm(acc) * np.linalg.norm(sun_to_sc))
    assert cos_angle == pytest.approx(1.0, abs=1e-9)
