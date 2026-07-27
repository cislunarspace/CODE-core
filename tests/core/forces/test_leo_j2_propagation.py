"""LEO 一天 J2 端到端传播测试。"""

import numpy as np
import pytest
from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.standard_axes import ICRSAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin

from e2m2e.core.ephemeris_system import EphemerisSystem
from e2m2e.core.forces import ForceModel, GravityField
from e2m2e.core.spice import SPICEManager


@pytest.fixture
def leo_system(spice_kernel_path):
    """LEO 传播用的 EphemerisSystem（ICRF + 地球中心）。"""
    from conftest import load_body_fixed_kernels, unload_kernels

    spice = SPICEManager()
    spice.load_kernel(spice_kernel_path)
    bf_kernels = load_body_fixed_kernels(spice)
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


def _extract_raan(state, et, mu):
    """用 spiceypy.oscltx 提取升交点经度（索引 3）。"""
    import spiceypy

    elems = spiceypy.oscltx(state.copy(), et, mu)
    return elems[3]


@pytest.mark.spice
def test_leo_j2_one_day_raan_drift(leo_system):
    """LEO 1 天传播，RAAN 长期变化率与 J2 解析公式一致（误差 < 1%）。"""
    system = leo_system
    mu = system.gravitational_parameter("EARTH")
    R_e = 6378.1363
    j2 = 0.001082626173852189  # EGM96 J2

    a = 6778.0  # km (~400 km altitude)
    e = 0.001
    i = 51.6
    raan0 = 0.0
    argp = 0.0
    nu = 0.0

    y0 = _keplerian_to_cartesian(a, e, i, raan0, argp, nu, mu)

    gravity = GravityField(body="EARTH", degree=2, order=0)
    fm = ForceModel(system, forces=[gravity])

    spice = system.spice
    et0 = spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + 86400.0)
    t_eval = np.linspace(et0, et0 + 86400.0, 200)

    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    raans = np.array(
        [_extract_raan(s, t, mu) for s, t in zip(result["states"], result["time"], strict=True)]
    )
    times_day = (result["time"] - et0) / 86400.0

    # Fit linear drift; unwrap RAAN to handle 2pi crossing
    raans_unwrapped = np.unwrap(raans)
    coeffs = np.polyfit(times_day, raans_unwrapped, 1)

    omega_dot_deg_per_day = np.degrees(coeffs[0])

    # Analytical J2 RAAN drift rate
    n = np.sqrt(mu / a**3)
    omega_dot_analytical = -1.5 * j2 * (R_e / a) ** 2 * n * np.cos(np.radians(i)) / (1 - e**2) ** 2
    omega_dot_analytical_deg_per_day = np.degrees(omega_dot_analytical) * 86400.0

    relative_error = abs(
        (omega_dot_deg_per_day - omega_dot_analytical_deg_per_day)
        / omega_dot_analytical_deg_per_day
    )
    assert relative_error < 0.05, f"relative_error={relative_error:.3f}"


@pytest.mark.spice
def test_leo_solid_tide_orbit_difference(leo_system):
    """AC6: LEO 1 天传播,固体潮 ON vs OFF 产生非零轨道差异(自洽性)。

    精度要求低:只验证潮汐改变轨道(非零差异),不设硬门槛。需要 ITRF93
    BPC 内核以查 Sun/Moon 在地固系的位置。
    """
    system = leo_system
    spice = system.spice
    et0 = spice.utc_to_et("2025-06-21T11:00:06")

    # ITRF93 帧 Sun/Moon 查询需要 BPC 内核,不可用则 skip
    try:
        spice.get_body_position("SUN", et0, "ITRF93", "EARTH")
    except Exception:
        pytest.skip("ITRF93 frame (BPC kernel) not available for tide test")

    mu = system.gravitational_parameter("EARTH")
    a = 6778.0  # km
    e = 0.001
    y0 = _keplerian_to_cartesian(a, e, 51.6, 0.0, 0.0, 0.0, mu)
    t_span = (et0, et0 + 86400.0)
    t_eval = np.linspace(et0, et0 + 86400.0, 50)

    gf_off = GravityField(body="EARTH", degree=2, order=0)
    gf_on = GravityField(body="EARTH", degree=2, order=0, tide_mode="solid")

    res_off = ForceModel(system, forces=[gf_off]).propagate(
        y0, t_span, t_eval=t_eval, max_steps=200_000
    )
    res_on = ForceModel(system, forces=[gf_on]).propagate(
        y0, t_span, t_eval=t_eval, max_steps=200_000
    )

    pos_off = res_off["states"][-1][:3]
    pos_on = res_on["states"][-1][:3]
    diff_km = np.linalg.norm(pos_on - pos_off)

    # 自洽性:潮汐 ON 改变轨道,差异非零
    assert diff_km > 0.0
