import numpy as np
import pytest

from e2m2e.algorithm.forces import (
    ForceModel,
    GravityField,
    SolarRadiationPressure,
    VariableMassFiniteBurn,
    VariableMassSolarRadiationPressure,
)
from tests.numerical.forces.conftest import (
    EARTH_RE,
    keplerian_to_cartesian,
    semi_major_axis,
)

pytestmark = [pytest.mark.force, pytest.mark.low_thrust]


def _build_lowthrust_problem(system, thrust, isp, mass, direction_kind="velocity"):
    """构造沿速度方向、可变质量的低推力 ForceModel。

    direction_kind:
        - "velocity": 固定方向向量无法跟随速度，故用 (t, state) -> v/|v| 可调用；
          可调用方向使 to_rust_spec 返回 None，走 Python 回退路径。
        - "fixed": 固定方向向量 [0,1,0]，to_rust_spec 非 None，走 Rust 7D 路径。
    """
    mu = system.gravitational_parameter("EARTH")
    r_earth = EARTH_RE
    a0 = r_earth + 300.0
    y0_6 = keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    if direction_kind == "velocity":

        def direction(_t: float, state: np.ndarray) -> np.ndarray:
            v = state[3:6]
            return v / np.linalg.norm(v)
    elif direction_kind == "fixed":
        direction = np.array([0.0, 1.0, 0.0])  # 与圆轨道初速方向一致
    else:
        raise ValueError(direction_kind)

    burn = VariableMassFiniteBurn(thrust=thrust, isp=isp, initial_mass=mass, direction=direction)
    gravity = GravityField(body="EARTH", degree=0, order=0)
    fm = ForceModel(system, forces=[gravity, burn])
    y0 = np.concatenate([y0_6, [mass]])
    return fm, y0


@pytest.mark.spice
def test_variable_mass_thrust_semi_major_axis_rate(earth_icrf_system):
    """可变质量低推力圆轨道提升：半长轴变化率与解析公式误差 < 5%。"""
    system = earth_icrf_system
    mu = system.gravitational_parameter("EARTH")

    thrust = 0.1  # N
    mass = 1000.0  # kg
    isp = 3000.0  # s
    duration_s = 1.0 * 86400.0

    # 固定方向（圆轨道初速沿 y），走 Rust 7D 路径
    fm, y0 = _build_lowthrust_problem(system, thrust, isp, mass, direction_kind="fixed")

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.linspace(et0, et0 + duration_s, 100)

    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    a_final = semi_major_axis(result["states"][-1], mu)
    a_theory = (y0[0] ** 1.5 + 1.5 * np.sqrt(mu) * (thrust / mass) * duration_s) ** (2.0 / 3.0)

    relative_error = abs((a_final - a_theory) / a_theory)
    assert relative_error < 0.05, (
        f"半长轴变化率偏差过大: measured={a_final:.3f} km, "
        f"theory={a_theory:.3f} km, error={relative_error:.1%}"
    )


@pytest.mark.spice
def test_variable_mass_consumption_matches_analytic(earth_icrf_system):
    """质量消耗对标解析值 Δm = −T·t_f/(Isp·g0)，误差 < 1e-6 kg。"""
    system = earth_icrf_system

    thrust = 0.5  # N
    mass = 1000.0  # kg
    isp = 3000.0  # s
    duration_s = 0.5 * 86400.0  # 12 h

    fm, y0 = _build_lowthrust_problem(system, thrust, isp, mass, direction_kind="fixed")

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.array([et0, et0 + duration_s])

    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    g0 = 9.81
    expected_mass = mass - thrust * duration_s / (isp * g0)
    actual_mass = float(result["states"][-1][6])
    assert abs(actual_mass - expected_mass) < 1e-6, (
        f"质量消耗不符: measured={actual_mass:.9f} kg, "
        f"theory={expected_mass:.9f} kg, diff={actual_mass - expected_mass:.2e}"
    )


@pytest.mark.spice
def test_variable_mass_uses_rust_path(earth_icrf_system):
    """固定方向的 VariableMassFiniteBurn 走 Rust 7D 路径（结果含 n_steps 键）。"""
    system = earth_icrf_system

    thrust = 0.1
    mass = 1000.0
    isp = 3000.0
    duration_s = 0.25 * 86400.0

    fm, y0 = _build_lowthrust_problem(system, thrust, isp, mass, direction_kind="fixed")
    assert fm._has_variable_mass_thrust()

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.array([et0, et0 + duration_s])

    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    # Rust 7D 路径返回 n_steps/n_rejected；Python 回退路径不返回这两个键
    assert "n_steps" in result, "应走 Rust 7D 路径（结果含 n_steps）"
    assert result["states"].shape[1] == 7, f"状态应为 7D, got {result['states'].shape}"
    assert result["n_steps"] > 0


@pytest.mark.spice
def test_variable_mass_callable_direction_rust_rejects_unsupported_force(earth_icrf_system):
    """Rust 不支持可调用方向时显式拒绝，不隐式回退 Python。"""
    system = earth_icrf_system

    thrust = 0.1
    mass = 1000.0
    isp = 3000.0
    duration_s = 0.25 * 86400.0

    fm, y0 = _build_lowthrust_problem(system, thrust, isp, mass, direction_kind="velocity")
    # 可调用方向不能序列化为 Rust force spec，必须显式拒绝。
    burn = next(e.force for e in fm._entries if isinstance(e.force, VariableMassFiniteBurn))
    assert burn.to_rust_spec(system) is None

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.array([et0, et0 + duration_s])

    with pytest.raises(NotImplementedError, match="callable direction.*Rust"):
        fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)


@pytest.mark.spice
def test_variable_mass_zero_thrust_no_fuel_consumption(earth_icrf_system):
    """零推力时质量守恒（边界情形）。"""
    system = earth_icrf_system
    mu = system.gravitational_parameter("EARTH")

    thrust = 0.0
    mass = 1000.0
    isp = 3000.0
    duration_s = 0.25 * 86400.0

    fm, y0 = _build_lowthrust_problem(system, thrust, isp, mass, direction_kind="fixed")

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.array([et0, et0 + duration_s])

    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    final_mass = float(result["states"][-1][6])
    assert abs(final_mass - mass) < 1e-9, (
        f"零推力下质量应守恒: measured={final_mass:.9f}, expected={mass}"
    )
    # 半长轴也应不变（纯二体圆轨道）
    a0 = semi_major_axis(y0, mu)
    a_final = semi_major_axis(result["states"][-1], mu)
    assert abs(a_final - a0) / a0 < 1e-6


def _build_srp_problem(system, srp_force, thrust):
    """构造含 SRP 的 7D 变质量传播问题（LEO 圆轨道，固定方向推力）。"""
    mu = system.gravitational_parameter("EARTH")
    a0 = EARTH_RE + 300.0
    y0_6 = keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)
    mass = 1000.0
    burn = VariableMassFiniteBurn(
        thrust=thrust,
        isp=3000.0,
        initial_mass=mass,
        direction=np.array([0.0, 1.0, 0.0]),
    )
    gravity = GravityField(body="EARTH", degree=0, order=0)
    fm = ForceModel(system, forces=[gravity, srp_force, burn])
    y0 = np.concatenate([y0_6, [mass]])
    return fm, y0


@pytest.mark.spice
def test_variable_mass_srp_matches_fixed_when_mass_constant(earth_icrf_system):
    """零推力质量恒定时，变质量 SRP 必须退化为同面质比的固定质量 SRP。"""
    system = earth_icrf_system
    area, mass, cr = 20.0, 1000.0, 1.3
    fm_var, y0 = _build_srp_problem(
        system, VariableMassSolarRadiationPressure(area=area, cr=cr), thrust=0.0
    )
    fm_fix, _ = _build_srp_problem(
        system, SolarRadiationPressure(area=area, mass=mass, cr=cr), thrust=0.0
    )

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + 86400.0)
    t_eval = np.array([et0 + 86400.0])

    state_var = fm_var.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)["states"][-1]
    state_fix = fm_fix.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)["states"][-1]

    diff_km = float(np.linalg.norm(state_var[:3] - state_fix[:3]))
    assert diff_km < 1e-9, f"质量恒定时两模型应逐位一致: diff={diff_km:.3e} km"


@pytest.mark.spice
def test_variable_mass_srp_mass_channel_active_under_burn(earth_icrf_system):
    """点火耗质量后，变质量 SRP 与取初始质量的固定 SRP 的轨迹应可分辨地分离。

    两天点火约耗 11.7 kg（约 1.2%），SRP 加速度同比增大；实测 LEO 两天弧
    轨迹差异约 9e-5 km（0.1 m 级，差异大部分被轨道相位吸收而非随 t²
    累积）。同物理零推力对照（上一测试）的差异 < 1e-9 km，故下限取
    1e-6 km 足以区分真实信号与数值噪声，上限 1 km 是 sanity 界。
    """
    system = earth_icrf_system
    area, mass, cr = 20.0, 1000.0, 1.3
    thrust = 2.0  # N
    duration_s = 2.0 * 86400.0

    fm_var, y0 = _build_srp_problem(
        system, VariableMassSolarRadiationPressure(area=area, cr=cr), thrust=thrust
    )
    fm_fix, _ = _build_srp_problem(
        system, SolarRadiationPressure(area=area, mass=mass, cr=cr), thrust=thrust
    )

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.array([et0 + duration_s])

    state_var = fm_var.propagate(y0, t_span, t_eval=t_eval, max_steps=400_000)["states"][-1]
    state_fix = fm_fix.propagate(y0, t_span, t_eval=t_eval, max_steps=400_000)["states"][-1]

    diff_km = float(np.linalg.norm(state_var[:3] - state_fix[:3]))
    assert state_var[6] < mass, "点火两天质量应下降"
    assert 1e-6 < diff_km < 1.0, f"质量通道未生效或差异异常: diff={diff_km:.3e} km"
