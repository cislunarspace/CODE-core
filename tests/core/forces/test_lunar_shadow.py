"""月影测试 —— 验收 #5：圆锥阴影模型支持地球 + 月球双遮挡体。

用真实 SPICE 星历（de440s）构造 cislunar 场景：SC 置于月球反日侧低轨，月球
应投下本影。三联断言锁定"阴影源是月球而非地球"，从而用真实几何（非合成
factor）验证多遮挡体路径端到端可运行。

References:
    - GMAT R2026a ``ShadowState`` / ``SolarRadiationPressure::GetShadowStateFromAllBodies``
    - GMAT GMT-6543 多遮挡体合成规范
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.ephemeris_system import EphemerisSystem
from e2m2e.core.forces.shadow import ConicalShadowModel
from e2m2e.core.spice import SPICEManager
from e2m2e.core.standard_axes import ICRSAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin

_MOON_R_KM = 1737.4


@pytest.fixture
def earth_icrf_system(spice_kernel_path):
    """地球中心 ICRF 传播系统（加载地月日三星历）。"""
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


def _anti_sun_sc_near_moon(system, et: float, alt_km: float = 100.0) -> np.ndarray:
    """月球反日侧、月面高度 ``alt_km`` 处的 SC 状态（地心 ICRF）。

    SC 置于月球 → 太阳连线的反方向，距月心 ``R_moon + alt_km``。此点处于月球
    本影锥内（月球视角径 ≫ 太阳视角径）。速度取月球地心速度（shadow 仅用位置）。
    """
    spice = system.spice
    moon = spice.get_body_state("MOON", et, "J2000", "EARTH")
    sun = spice.get_body_state("SUN", et, "J2000", "EARTH")[:3]
    moon_pos = moon[:3]
    moon_vel = moon[3:]
    sun_from_moon = sun - moon_pos
    u_sun = sun_from_moon / np.linalg.norm(sun_from_moon)  # Moon → Sun 单位向量
    sc_pos = moon_pos - u_sun * (_MOON_R_KM + alt_km)  # 反日侧（远离太阳）
    return np.concatenate([sc_pos, moon_vel])


@pytest.mark.spice
def test_lunar_umbra_shadow_comes_from_moon_not_earth(earth_icrf_system) -> None:
    """月球反日侧低轨 SC 应被月球本影遮挡，且该处地球不遮挡。

    三联断言用真实星历几何证明"月影"：
    1. ``bodies=["MOON"]`` → flux ≈ 0（月球投下本影）；
    2. ``bodies=["EARTH"]`` → flux ≈ 1（该处地球不在 SC-太阳线上，不遮挡——
       从而排除"阴影其实来自地球"的可能）；
    3. ``bodies=["EARTH", "MOON"]`` → flux ≈ 0（GMT-6543 合成：任一本影→0，
       多遮挡体循环 + 合成在真实双体几何下端到端运行）。

    历元取 2025-09-22，远离 2025-09-07 月全食，确保月球不在地影中。
    """
    system = earth_icrf_system
    spice = system.spice
    et = spice.utc_to_et("2025-09-22T12:00:00")
    state = _anti_sun_sc_near_moon(system, et, alt_km=100.0)

    flux_moon = ConicalShadowModel(bodies=["MOON"]).flux_factor(et, state, system)
    flux_earth = ConicalShadowModel(bodies=["EARTH"]).flux_factor(et, state, system)
    flux_both = ConicalShadowModel(bodies=["EARTH", "MOON"]).flux_factor(
        et, state, system
    )

    assert flux_moon < 1e-6, f"月球应投本影 flux≈0，实际 {flux_moon:.3e}"
    assert flux_earth > 0.99, (
        f"该处地球不应遮挡 flux≈1（证明阴影源是月球），实际 {flux_earth:.4f}"
    )
    assert flux_both < 1e-6, (
        f"地+月合成应为本影 flux≈0（GMT-6543 任一本影→0），实际 {flux_both:.3e}"
    )
