"""SRP / Shadow 系统感知路径集成测试（需要 SPICE 内核）。

验证 ``compute_acceleration`` / ``flux_factor`` 通过 ``system`` 取太阳与遮挡体
位置后，与手动（SPICE 取位 + 纯函数）计算一致。范式同 ``test_drag_transform.py``。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.ephemeris_system import EphemerisSystem
from e2m2e.core.forces.shadow import ConicalShadowModel
from e2m2e.core.forces.srp import SolarRadiationPressure
from e2m2e.core.spice import SPICEManager
from e2m2e.core.standard_axes import ICRSAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin

_R_EARTH = 6378.1363
_R_SUN = 695700.0


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
def test_srp_compute_acceleration_matches_manual(earth_icrf_system) -> None:
    """compute_acceleration(t, state, system) 与手算一致（SPICE 取日位 + 纯函数）。"""
    system = earth_icrf_system
    et = system.spice.utc_to_et("2025-06-21T11:00:06")
    state = np.array([6778.0, 0.0, 0.0, 0.0, 7.7, 0.0])

    shadow = ConicalShadowModel(bodies=["EARTH"])
    srp = SolarRadiationPressure(area=10.0, mass=1000.0, cr=1.5, shadow=shadow)

    acc_system = srp.compute_acceleration(et, state, system)

    # 手算
    sun_pos = _sun_pos_rel_earth(system, et)
    sun_to_sc = state[:3] - sun_pos
    flux = shadow.flux_factor(et, state, system)
    acc_manual = srp._compute_srp_acceleration(sun_to_sc, flux)
    np.testing.assert_allclose(acc_system, acc_manual, rtol=1e-12)


@pytest.mark.spice
def test_srp_acceleration_points_away_from_sun(earth_icrf_system) -> None:
    """SRP 加速度在 ICRF 中指向远离太阳（与 Sun→SC 同向）。"""
    system = earth_icrf_system
    et = system.spice.utc_to_et("2025-06-21T11:00:06")
    state = np.array([6778.0, 0.0, 0.0, 0.0, 7.7, 0.0])

    srp = SolarRadiationPressure(area=10.0, mass=1000.0, cr=1.5)  # 无阴影
    acc = srp.compute_acceleration(et, state, system)

    sun_pos = _sun_pos_rel_earth(system, et)
    sun_to_sc = state[:3] - sun_pos
    cos_angle = np.dot(acc, sun_to_sc) / (
        np.linalg.norm(acc) * np.linalg.norm(sun_to_sc)
    )
    assert cos_angle == pytest.approx(1.0, abs=1e-9)
