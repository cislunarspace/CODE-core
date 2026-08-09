"""低推力传播端到端验证。

验证半长轴变化率与解析公式误差 < 5%、7 天螺旋轨道演化、
以及 FiniteBurn 配置 round-trip。
"""

import numpy as np
import pytest

from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
from e2m2e.algorithm.dynamics.ephemeris_system import EphemerisSystem
from e2m2e.algorithm.forces import FiniteBurn, ForceModel, GravityField
from e2m2e.data.kernels.manager import SPICEManager

pytestmark = pytest.mark.force


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


def _eccentricity(state, mu):
    """从状态向量计算偏心率。"""
    r_vec = state[:3]
    v_vec = state[3:6]
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    rv_dot = np.dot(r_vec, v_vec)
    e_vec = ((v**2 - mu / r) * r_vec - rv_dot * v_vec) / mu
    return float(np.linalg.norm(e_vec))


@pytest.fixture
def earth_ephemeris_system(spice_kernel_path):
    """Earth-centered J2000 ephemeris system for low-thrust tests."""
    from kernel_helpers import load_body_fixed_kernels, unload_kernels

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


@pytest.mark.spice
def test_low_thrust_circular_orbit_semi_major_axis_rate(earth_ephemeris_system):
    """低推力圆轨道提升：半长轴变化率与解析公式误差 < 5%。"""
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    r_earth = 6378.137
    a0 = r_earth + 300.0
    y0 = _keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    thrust = 0.1  # N
    mass = 1000.0  # kg
    duration_s = 1.0 * 86400.0

    def thrust_profile(_t: float) -> float:
        return thrust

    def direction(_t: float, state: np.ndarray) -> np.ndarray:
        v = state[3:6]
        return v / np.linalg.norm(v)

    burn = FiniteBurn(thrust_profile=thrust_profile, direction=direction, mass=mass)
    gravity = GravityField(body="EARTH", degree=2, order=0)
    fm = ForceModel(system, forces=[gravity, burn])

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.linspace(et0, et0 + duration_s, 100)

    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    a_history = np.array([_semi_major_axis(s, mu) for s in result["states"]])
    a_final = a_history[-1]

    # 解析解：a(t) = (a0^(3/2) + 3/2 * sqrt(mu) * (F/m) * t)^(2/3)
    a_theory = (a0**1.5 + 1.5 * np.sqrt(mu) * (thrust / mass) * duration_s) ** (2.0 / 3.0)

    relative_error = abs((a_final - a_theory) / a_theory)
    assert relative_error < 0.05, (
        f"半长轴变化率偏差过大: measured={a_final:.3f} km, "
        f"theory={a_theory:.3f} km, error={relative_error:.1%}"
    )


@pytest.mark.slow
@pytest.mark.spice
def test_low_thrust_spiral_orbit_evolution(earth_ephemeris_system):
    """7 天低推力螺旋轨道：半长轴单调提升，偏心率保持低值。"""
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    r_earth = 6378.137
    a0 = r_earth + 300.0
    y0 = _keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    thrust = 0.1  # N
    mass = 1000.0  # kg
    duration_s = 7.0 * 86400.0

    def thrust_profile(_t: float) -> float:
        return thrust

    def direction(_t: float, state: np.ndarray) -> np.ndarray:
        v = state[3:6]
        return v / np.linalg.norm(v)

    burn = FiniteBurn(thrust_profile=thrust_profile, direction=direction, mass=mass)
    gravity = GravityField(body="EARTH", degree=0, order=0)
    fm = ForceModel(system, forces=[gravity, burn])
    # 长弧段传播放宽容差与最大步长，减少积分步数
    fm.rtol = 1e-10
    fm.atol = 1e-10
    fm.max_step = 600.0

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.linspace(et0, et0 + duration_s, 50)

    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=1_000_000)

    a_history = np.array([_semi_major_axis(s, mu) for s in result["states"]])
    e_history = np.array([_eccentricity(s, mu) for s in result["states"]])

    # 半长轴显著提升（>5 km）且线性拟合斜率为正
    assert a_history[-1] > a0 + 5.0, f"半长轴应提升超过 5 km, got {a_history[-1] - a0:.3f} km"
    times_day = (result["time"] - result["time"][0]) / 86400.0
    slope = np.polyfit(times_day, a_history, 1)[0]
    assert slope > 0.0, f"半长轴 secular 斜率应为正, got {slope:.3f} km/day"

    # 偏心率保持低值
    assert e_history[-1] < 0.05, f"最终偏心率应 < 0.05, got {e_history[-1]:.6f}"
    assert np.max(e_history) < 0.05, f"最大偏心率应 < 0.05, got {np.max(e_history):.6f}"


@pytest.mark.spice
def test_low_thrust_config_round_trip(earth_ephemeris_system):
    """FiniteBurn 配置往返后，低推力传播结果一致。"""
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    r_earth = 6378.137
    a0 = r_earth + 300.0
    y0 = _keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    thrust = 0.1
    mass = 1000.0
    duration_s = 1.0 * 86400.0

    from e2m2e.algorithm.forces.force_config import build_force

    burn_original = FiniteBurn(
        thrust_profile=build_force(
            "FiniteBurn",
            {
                "thrust_profile": {"kind": "constant", "thrust": thrust},
                "direction": {"kind": "fixed", "vector": [0.0, 1.0, 0.0]},
                "mass": mass,
            },
        ).thrust_profile,
        direction=[0.0, 1.0, 0.0],
        mass=mass,
    )
    fm_original = ForceModel(
        system,
        forces=[GravityField(body="EARTH", degree=0, order=0), burn_original],
    )

    config = fm_original.to_config()
    fm_restored = ForceModel.from_config(config, system)

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.array([et0, et0 + duration_s])

    result_original = fm_original.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)
    result_restored = fm_restored.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    np.testing.assert_allclose(
        result_original["states"][-1],
        result_restored["states"][-1],
        rtol=1e-12,
        atol=1e-12,
    )
