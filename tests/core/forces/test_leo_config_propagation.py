"""LEO 配置构建 vs 手动构建传播一致性测试（需 SPICE 内核）。"""

import numpy as np
import pytest
from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.standard_axes import ICRSAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin

from e2m2e.core.atmosphere import ExponentialAtmosphere
from e2m2e.core.ephemeris_system import EphemerisSystem
from e2m2e.core.forces import DragModel, ForceModel, GravityField
from e2m2e.core.spice import SPICEManager

_EARTH_R_KM = 6378.137
_MU_EARTH = 398600.4415  # km³/s²


@pytest.fixture
def leo_system(spice_kernel_path):
    """LEO 传播用的 EphemerisSystem（ICRF + 地球中心）。"""
    from conftest import load_body_fixed_kernels, unload_kernels

    spice = SPICEManager()
    spice.load_kernel(spice_kernel_path)
    bf_kernels = load_body_fixed_kernels(spice)
    try:
        system = EphemerisSystem(bodies=["EARTH"], spice=spice, origin="EARTH")
        system.coordinate_system = CoordinateSystem(
            axes=ICRSAxes(),
            origin=CelestialBodyOrigin(body="EARTH", spice=spice),
        )
        yield system
    finally:
        unload_kernels(spice, bf_kernels)
        spice.unload_kernel(spice_kernel_path)


def _keplerian_to_cartesian(a, e, i, raan, argp, nu, mu):
    """将开普勒根数转为笛卡尔状态。"""
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))
    i = np.radians(i)
    raan = np.radians(raan)
    argp = np.radians(argp)
    nu = np.radians(nu)
    r_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pqw = np.array(
        [
            -np.sqrt(mu / p) * np.sin(nu),
            np.sqrt(mu / p) * (e + np.cos(nu)),
            0.0,
        ]
    )
    R3_raan = np.array(
        [[np.cos(raan), -np.sin(raan), 0.0], [np.sin(raan), np.cos(raan), 0.0], [0.0, 0.0, 1.0]]
    )
    R1_i = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(i), -np.sin(i)], [0.0, np.sin(i), np.cos(i)]])
    R3_argp = np.array(
        [[np.cos(argp), -np.sin(argp), 0.0], [np.sin(argp), np.cos(argp), 0.0], [0.0, 0.0, 1.0]]
    )
    R = R3_raan @ R1_i @ R3_argp
    return np.concatenate([R @ r_pqw, R @ v_pqw])


@pytest.mark.spice
def test_leo_config_vs_manual_propagation_match(leo_system):
    """to_config/from_config 重建的力模型传播轨迹与手动构建一致（< 1e-12）。"""
    system = leo_system

    # 手动构建 LEO 力模型（J2 + 阻力）
    fm_manual = ForceModel(system)
    fm_manual.add_force(GravityField("EARTH", degree=2, order=0), name="j2")
    fm_manual.add_force(
        DragModel(atmosphere=ExponentialAtmosphere(), area=10.0, mass=1000.0, cd=2.2),
        name="drag",
    )

    # 经 config 重建
    config = ForceModel.to_config(fm_manual)
    fm_config = ForceModel.from_config(config, system)

    # 同一初始状态与时间区间传播
    a0 = _EARTH_R_KM + 400.0
    y0 = _keplerian_to_cartesian(a0, 0.001, 51.6, 0.0, 0.0, 0.0, _MU_EARTH)
    spice = system.spice
    et0 = spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + 600.0)
    t_eval = np.linspace(et0, et0 + 600.0, 50)

    result_manual = fm_manual.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)
    result_config = fm_config.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    np.testing.assert_allclose(result_manual["states"], result_config["states"], atol=1e-12)
