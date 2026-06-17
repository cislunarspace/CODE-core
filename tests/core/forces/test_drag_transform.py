"""DragModel ITRF↔传播系转换集成测试（需 SPICE 内核）。

验证内部转换与手动转换结果一致、旋转保模。
"""

import numpy as np
import pytest

from e2m2e.core.atmosphere import ExponentialAtmosphere
from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.ephemeris_system import EphemerisSystem
from e2m2e.core.forces.drag import DragModel
from e2m2e.core.spice import SPICEManager
from e2m2e.core.standard_axes import ICRSAxes, ITRFApproxAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin

_EARTH_R_KM = 6378.137
_KM_TO_M = 1000.0


@pytest.fixture
def earth_icrf_system(spice_kernel_path):
    """地球中心 ICRF 传播系统。"""
    spice = SPICEManager()
    spice.load_kernel(spice_kernel_path)
    try:
        system = EphemerisSystem(
            bodies=["EARTH"],
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


@pytest.mark.spice
def test_drag_transform_matches_manual(earth_icrf_system):
    """DragModel 内部 ITRF 转换与手动转换结果一致。

    手动：ICRF 状态 → ITRF → 算阻力 → 转回 ICRF。
    DragModel 应内部完成同样的流程。
    """
    system = earth_icrf_system
    spice = system.spice
    et = spice.utc_to_et("2025-06-21T11:00:06")

    # ICRF 状态：400 km 圆轨道（近似）
    state_icrf = np.array([6778.0, 0.0, 0.0, 0.0, 7.7, 0.0])

    atm = ExponentialAtmosphere()
    drag = DragModel(atmosphere=atm, area=10.0, mass=1000.0, cd=2.2)

    # DragModel 内部完成 ICRF→ITRF→算阻力→转回 ICRF
    acc_drag = drag.compute_acceleration(et, state_icrf, system)

    # 手动流程：构造 ITRF CS，转换状态，算阻力，转换回
    itrf_cs = CoordinateSystem(
        axes=ITRFApproxAxes(),
        origin=CelestialBodyOrigin(body="EARTH", spice=spice),
    )
    state_itrf = system.coordinate_system.transform_state(
        state_icrf, from_cs=system.coordinate_system, to_cs=itrf_cs, et=et
    )
    acc_itrf = drag._compute_drag_in_itrf(state_itrf[:3], state_itrf[3:6])
    acc_manual = system.coordinate_system.transform_vector(
        acc_itrf, from_cs=itrf_cs, to_cs=system.coordinate_system, et=et
    )

    np.testing.assert_allclose(acc_drag, acc_manual, rtol=1e-12)


@pytest.mark.spice
def test_drag_magnitude_preserved_under_rotation(earth_icrf_system):
    """ICRF 中计算的阻力加速度量级与直接 ITRF 计算一致（旋转保模）。"""
    system = earth_icrf_system
    spice = system.spice
    et = spice.utc_to_et("2025-06-21T11:00:06")

    state_icrf = np.array([6778.0, 0.0, 0.0, 0.0, 7.7, 0.0])
    atm = ExponentialAtmosphere()
    drag = DragModel(atmosphere=atm, area=10.0, mass=1000.0, cd=2.2)

    # 通过 system 计算的 ICRF 阻力
    acc_icrf = drag.compute_acceleration(et, state_icrf, system)

    # 转到 ITRF 后直接计算
    itrf_cs = CoordinateSystem(
        axes=ITRFApproxAxes(),
        origin=CelestialBodyOrigin(body="EARTH", spice=spice),
    )
    state_itrf = system.coordinate_system.transform_state(
        state_icrf, from_cs=system.coordinate_system, to_cs=itrf_cs, et=et
    )
    acc_itrf = drag._compute_drag_in_itrf(state_itrf[:3], state_itrf[3:6])

    # 旋转保模：ICRF 和 ITRF 中的阻力模长相等
    np.testing.assert_allclose(np.linalg.norm(acc_icrf), np.linalg.norm(acc_itrf), rtol=1e-10)
