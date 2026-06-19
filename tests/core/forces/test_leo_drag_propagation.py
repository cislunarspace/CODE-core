"""LEO 阻力传播端到端测试（需 SPICE 内核）。"""

import numpy as np
import pytest

from e2m2e.core.atmosphere import ExponentialAtmosphere
from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.ephemeris_system import EphemerisSystem
from e2m2e.core.forces import ForceModel, GravityField
from e2m2e.core.forces.drag import DragModel
from e2m2e.core.spice import SPICEManager
from e2m2e.core.standard_axes import ICRSAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin

_EARTH_R_KM = 6378.137
_MU_EARTH = 398600.4415  # km³/s²


@pytest.fixture
def leo_system(spice_kernel_path):
    """LEO 传播用的 EphemerisSystem（ICRF + 地球中心）。"""
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
        [
            [np.cos(raan), -np.sin(raan), 0.0],
            [np.sin(raan), np.cos(raan), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R1_i = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(i), -np.sin(i)],
            [0.0, np.sin(i), np.cos(i)],
        ]
    )
    R3_argp = np.array(
        [
            [np.cos(argp), -np.sin(argp), 0.0],
            [np.sin(argp), np.cos(argp), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R = R3_raan @ R1_i @ R3_argp

    r_eci = R @ r_pqw
    v_eci = R @ v_pqw
    return np.concatenate([r_eci, v_eci])


def _semi_major_axis(state, mu):
    """从状态向量用能量公式计算半长轴。"""
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:6])
    energy = v**2 / 2.0 - mu / r
    return -mu / (2.0 * energy)


@pytest.mark.spice
def test_leo_drag_semi_major_axis_decays(leo_system):
    """LEO 1 天传播（J2 + 阻力），半长轴衰减方向正确且量级合理。"""
    system = leo_system
    mu = _MU_EARTH

    a0 = _EARTH_R_KM + 300.0  # 300 km 圆轨道
    e = 0.001
    i = 51.6

    y0 = _keplerian_to_cartesian(a0, e, i, 0.0, 0.0, 0.0, mu)

    atm = ExponentialAtmosphere()
    drag = DragModel(atmosphere=atm, area=10.0, mass=1000.0, cd=2.2)
    gravity = GravityField(body="EARTH", degree=2, order=0)
    fm = ForceModel(system, forces=[gravity, drag])

    spice = system.spice
    et0 = spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + 86400.0)
    t_eval = np.linspace(et0, et0 + 86400.0, 200)

    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    # Osculating 半长轴随轨道相位振荡（偏心率 + 摄动），振幅 ~9 km，
    # 远大于日衰减 ~2 km。对 a(t) 做线性拟合提取 secular 斜率。
    a_history = np.array([_semi_major_axis(s, mu) for s in result["states"]])
    times_day = (result["time"] - result["time"][0]) / 86400.0
    coeffs = np.polyfit(times_day, a_history, 1)
    delta_a_per_day = coeffs[0]  # km/day

    # 方向：半长轴缩短
    assert delta_a_per_day < 0, f"半长轴应衰减，但 da/day={delta_a_per_day:.4f} km/day"

    # 量级：与解析 da/dt = -rho·BC·sqrt(mu·a) 交叉验证（±50%）
    rho = atm.density(300.0)  # kg/m³
    bc = 2.2 * 10.0 / 1000.0  # m²/kg
    mu_si = _MU_EARTH * 1e9  # m³/s²
    a_si = a0 * 1000  # m
    da_dt_theory = -rho * bc * np.sqrt(mu_si * a_si)  # m/s
    delta_a_theory = da_dt_theory * 86400.0 / 1000.0  # km/day

    relative_error = abs((delta_a_per_day - delta_a_theory) / delta_a_theory)
    assert relative_error < 0.50, (
        f"半长轴衰减量级偏差过大: measured={delta_a_per_day:.4f} km/day, "
        f"theory={delta_a_theory:.4f} km/day, error={relative_error:.1%}"
    )
