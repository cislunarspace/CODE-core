"""相对论修正力模型测试。

覆盖有限加速度、Schwarzschild 开关、配置 round-trip 与三项独立启用。
"""

import types

import numpy as np
import pytest

from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.ephemeris_system import EphemerisSystem
from e2m2e.core.forces import ForceModel, GravityField, RelativisticCorrection
from e2m2e.core.spice import SPICEManager
from e2m2e.core.standard_axes import ICRSAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin


def _make_system(mu: float = 398600.435507):
    """返回一个只提供 gravitational_parameter 的最小 system stub。"""

    def get_body_state(target, et, frame, observer):
        # 返回一个假想的地球绕太阳状态：1 AU x 方向，30 km/s y 方向
        if target.upper() == "EARTH":
            return np.array([1.496e8, 0.0, 0.0, 0.0, 29.78, 0.0])
        if target.upper() == "SUN":
            return np.zeros(6)
        raise ValueError(f"unknown body {target}")

    spice = types.SimpleNamespace(get_body_state=get_body_state)
    return types.SimpleNamespace(
        gravitational_parameter=lambda _body: mu,
        spice=spice,
    )


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


@pytest.fixture
def earth_ephemeris_system(spice_kernel_path):
    """Earth-centered J2000 ephemeris system for relativistic tests."""
    spice = SPICEManager()
    spice.load_kernel(spice_kernel_path)
    try:
        system = EphemerisSystem(
            bodies=["EARTH", "SUN"],
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


# 地球角动量矢量近似值 (km²/s)，与 GMAT 自动计算结果同量级。
_EARTH_ANGULAR_MOMENTUM = np.array([0.0, 0.0, 1.18e3])


def test_relativistic_correction_returns_finite_acceleration():
    """Tracer bullet: RelativisticCorrection 能返回一个有限小的三维加速度。"""
    force = RelativisticCorrection(
        central_body="EARTH",
        angular_momentum_vector=[0.0, 0.0, 7.5e33],
    )

    system = _make_system()
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    acc = force.compute_acceleration(0.0, state, system)

    assert acc.shape == (3,)
    assert np.all(np.isfinite(acc))


def test_schwarzschild_switch_controls_acceleration():
    """Schwarzschild 开关能控制该项加速度是否为零。"""
    system = _make_system()
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])

    force_on = RelativisticCorrection(
        central_body="EARTH",
        enable_lense_thirring=False,
        enable_de_sitter=False,
    )
    acc_on = force_on.compute_acceleration(0.0, state, system)

    force_off = RelativisticCorrection(
        central_body="EARTH",
        enable_schwarzschild=False,
        enable_lense_thirring=False,
        enable_de_sitter=False,
    )
    acc_off = force_off.compute_acceleration(0.0, state, system)

    assert np.linalg.norm(acc_on) > 0.0
    np.testing.assert_array_equal(acc_off, np.zeros(3))


def test_config_round_trip():
    """RelativisticCorrection 支持 ForceModel 配置往返。"""
    from e2m2e.core.forces.force_config import build_force, serialize_force

    original = RelativisticCorrection(
        central_body="Earth",
        primary_body="Sun",
        enable_schwarzschild=True,
        enable_lense_thirring=False,
        enable_de_sitter=True,
        angular_momentum_vector=[0.0, 0.0, 7.5e33],
        body_radius=6378.137,
        c=299792.458,
        gamma=1.0,
    )

    config = serialize_force(original)
    restored = build_force(config["type"], config["params"])

    assert isinstance(restored, RelativisticCorrection)
    assert restored.central_body == "EARTH"
    assert restored.primary_body == "SUN"
    assert restored.enable_schwarzschild is True
    assert restored.enable_lense_thirring is False
    assert restored.enable_de_sitter is True
    np.testing.assert_array_equal(
        restored.angular_momentum_vector, np.array([0.0, 0.0, 7.5e33])
    )
    assert restored.body_radius == pytest.approx(6378.137)
    assert restored.c == pytest.approx(299792.458)
    assert restored.gamma == pytest.approx(1.0)


@pytest.mark.parametrize(
    "enabled_term",
    ["schwarzschild", "lense_thirring", "de_sitter"],
)
def test_each_term_can_be_enabled_independently(enabled_term: str):
    """三项相对论效应可以独立启用并产生非零加速度。"""
    system = _make_system()
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])

    kwargs = {
        "central_body": "EARTH",
        "enable_schwarzschild": False,
        "enable_lense_thirring": False,
        "enable_de_sitter": False,
        "angular_momentum_vector": [0.0, 0.0, 7.5e33],
    }
    kwargs[f"enable_{enabled_term}"] = True

    force = RelativisticCorrection(**kwargs)
    acc = force.compute_acceleration(0.0, state, system)
    assert np.linalg.norm(acc) > 0.0, f"{enabled_term} 启用后应产生非零加速度"


def test_automatic_angular_momentum_raises_without_kernels():
    """未提供角动量覆盖值且无法自动计算时，应抛出 RelativisticCorrectionError。"""
    system = _make_system()
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])

    force = RelativisticCorrection(
        central_body="EARTH",
        enable_schwarzschild=False,
        enable_lense_thirring=True,
        enable_de_sitter=False,
    )

    with pytest.raises(Exception, match="."):  # 当前为 RelativisticCorrectionError
        force.compute_acceleration(0.0, state, system)


@pytest.mark.spice
def test_gps_relativistic_position_difference_magnitude(earth_ephemeris_system):
    """GPS 轨道 1 天传播，相对论修正导致可观测的终端位置漂移。"""
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    # GPS 类轨道
    a0 = 26560.0
    y0 = _keplerian_to_cartesian(a0, 0.0, 55.0, 0.0, 0.0, 0.0, mu)

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + 86400.0)
    t_eval = np.array([et0, et0 + 86400.0])

    gravity = GravityField(body="EARTH", degree=2, order=0)
    fm_without = ForceModel(system, forces=[gravity])
    result_without = fm_without.propagate(
        y0, t_span, t_eval=t_eval, max_steps=200_000
    )

    relcorr = RelativisticCorrection(
        central_body="EARTH",
        angular_momentum_vector=_EARTH_ANGULAR_MOMENTUM,
    )
    fm_with = ForceModel(system, forces=[gravity, relcorr])
    result_with = fm_with.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    pos_diff = np.linalg.norm(
        result_with["states"][-1, :3] - result_without["states"][-1, :3]
    )
    # 物理量级：GPS 轨道 Schwarzschild 修正约 0.3 mm/天（3e-7 km）。
    # 下界收到 1/10 物理量级以防回归把数量级改坏（远低于物理 3 个数量级 → 收紧到 0.1×）。
    # 上界 0.01 km（10 cm/天）覆盖 Lense-Thirring / de Sitter 等次级项贡献。
    assert 3e-8 <= pos_diff <= 0.01, f"GPS 1-day position diff = {pos_diff:.6e} km"


@pytest.mark.spice
def test_leo_relativistic_position_difference_magnitude(earth_ephemeris_system):
    """LEO 轨道 1 天传播，相对论修正导致可观测的终端位置漂移。"""
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    r_earth = 6378.137
    a0 = r_earth + 400.0
    y0 = _keplerian_to_cartesian(a0, 0.0, 51.6, 0.0, 0.0, 0.0, mu)

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + 86400.0)
    t_eval = np.array([et0, et0 + 86400.0])

    gravity = GravityField(body="EARTH", degree=2, order=0)
    fm_without = ForceModel(system, forces=[gravity])
    result_without = fm_without.propagate(
        y0, t_span, t_eval=t_eval, max_steps=200_000
    )

    relcorr = RelativisticCorrection(
        central_body="EARTH",
        angular_momentum_vector=_EARTH_ANGULAR_MOMENTUM,
    )
    fm_with = ForceModel(system, forces=[gravity, relcorr])
    result_with = fm_with.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    pos_diff = np.linalg.norm(
        result_with["states"][-1, :3] - result_without["states"][-1, :3]
    )
    # 物理量级：LEO Schwarzschild 修正约 2.5 mm/天（2.5e-6 km）。
    # 下界收到 1/10 物理量级以防回归把数量级改坏（原 1e-3 km 比物理宽 2.7 个数量级 → 收紧到 0.1×）。
    # 上界 0.01 km（10 cm/天）覆盖 Lense-Thirring / de Sitter 等次级项贡献。
    assert 2.5e-7 <= pos_diff <= 0.01, f"LEO 1-day position diff = {pos_diff:.6e} km"
