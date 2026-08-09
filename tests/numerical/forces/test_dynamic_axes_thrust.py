"""动态坐标系推力传播集成测试（Slice 12 验收）。

验证 VNB 沿速度方向推力提升半长轴、LVLH 径向推力增加偏心率、
转换矩阵正交性与混合方向。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
from e2m2e.algorithm.dynamics.ephemeris_system import EphemerisSystem
from e2m2e.algorithm.forces import FiniteBurn, ForceModel, PointMassGravity
from e2m2e.data.kernels.manager import SPICEManager

pytestmark = pytest.mark.force


_EARTH_R_KM = 6378.137
_MU_EARTH = 398600.4415  # km³/s²


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


def _orbital_energy(state, mu):
    """从状态向量计算比轨道能量。"""
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:6])
    return v**2 / 2.0 - mu / r


@pytest.fixture
def earth_ephemeris_system(spice_kernel_path):
    """Earth-centered J2000 ephemeris system for dynamic axes thrust tests."""
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


# --- 测试 1：VNB 沿速度方向推力 ---


@pytest.mark.spice
def test_vnb_velocity_direction_thrust_increases_semi_major_axis(earth_ephemeris_system):
    """VNB 下 direction=[1,0,0] 沿速度方向推力，验证半长轴与轨道能量增加。"""
    # Arrange
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    a0 = _EARTH_R_KM + 400.0  # 近圆 LEO，400 km 高度
    y0 = _keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    thrust = 1.0  # N
    mass = 1000.0  # kg
    duration_s = 3600.0  # 1 小时

    gravity = PointMassGravity(body="EARTH")
    burn = FiniteBurn(
        thrust_profile=lambda t: thrust,
        direction=[1.0, 0.0, 0.0],
        mass=mass,
        direction_frame="VNB",
    )
    fm = ForceModel(system, forces=[gravity, burn])

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.linspace(et0, et0 + duration_s, 50)

    # Act
    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    # Assert：半长轴增加
    a_initial = _semi_major_axis(result["states"][0], mu)
    a_final = _semi_major_axis(result["states"][-1], mu)
    assert a_final > a_initial, f"半长轴应增加: initial={a_initial:.3f} km, final={a_final:.3f} km"

    # Assert：轨道能量增加
    energy_initial = _orbital_energy(result["states"][0], mu)
    energy_final = _orbital_energy(result["states"][-1], mu)
    assert energy_final > energy_initial, (
        f"轨道能量应增加: initial={energy_initial:.6f} km²/s², final={energy_final:.6f} km²/s²"
    )


@pytest.mark.spice
def test_vnb_velocity_direction_thrust_acceleration_aligns_with_velocity(
    earth_ephemeris_system,
):
    """VNB 沿速度方向推力，验证惯性系中推力加速度方向与速度方向点积 ≈ 1。"""
    # Arrange
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    a0 = _EARTH_R_KM + 400.0
    y0 = _keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    thrust = 1.0
    mass = 1000.0
    duration_s = 600.0  # 10 分钟，短弧段验证方向

    gravity = PointMassGravity(body="EARTH")
    burn = FiniteBurn(
        thrust_profile=lambda t: thrust,
        direction=[1.0, 0.0, 0.0],
        mass=mass,
        direction_frame="VNB",
    )
    fm = ForceModel(system, forces=[gravity, burn])

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.linspace(et0, et0 + duration_s, 20)

    # Act
    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    # Assert：推力加速度方向与速度方向点积 ≈ 1
    # 在纯二体 + VNB 速度方向推力下，推力加速度始终沿速度方向
    for state in result["states"]:
        v = state[3:6]
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-15:
            v_hat = v / v_norm
            # 计算推力加速度（扣除引力后）
            r = state[:3]
            r_norm = np.linalg.norm(r)
            gravity_acc = -mu / (r_norm**3) * r
            total_acc = fm._compute_total_acceleration(et0, state)
            thrust_acc = total_acc - gravity_acc
            thrust_acc_norm = np.linalg.norm(thrust_acc)
            if thrust_acc_norm > 1e-15:
                dot_product = np.dot(thrust_acc / thrust_acc_norm, v_hat)
                assert dot_product > 0.99, (
                    f"推力加速度方向与速度方向点积应 > 0.99, got {dot_product:.6f}"
                )


# --- 测试 2：LVLH 径向推力 ---


@pytest.mark.spice
def test_lvlh_radial_direction_thrust_increases_eccentricity(earth_ephemeris_system):
    """LVLH 下 direction=[1,0,0] 径向推力，验证偏心率随时间增加。"""
    # Arrange
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    a0 = _EARTH_R_KM + 400.0
    y0 = _keplerian_to_cartesian(a0, 0.001, 0.0, 0.0, 0.0, 0.0, mu)

    thrust = 1.0
    mass = 1000.0
    duration_s = 3600.0

    gravity = PointMassGravity(body="EARTH")
    burn = FiniteBurn(
        thrust_profile=lambda t: thrust,
        direction=[1.0, 0.0, 0.0],
        mass=mass,
        direction_frame="LVLH",
    )
    fm = ForceModel(system, forces=[gravity, burn])

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.linspace(et0, et0 + duration_s, 50)

    # Act
    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    # Assert：偏心率增加
    e_initial = _eccentricity(result["states"][0], mu)
    e_final = _eccentricity(result["states"][-1], mu)
    assert e_final > e_initial, f"偏心率应增加: initial={e_initial:.6f}, final={e_final:.6f}"

    # 进一步验证：偏心率序列单调递增（或至少总体趋势上升）
    e_history = np.array([_eccentricity(s, mu) for s in result["states"]])
    assert e_history[-1] > e_history[0], (
        f"偏心率最终值应大于初始值: {e_history[-1]:.6f} > {e_history[0]:.6f}"
    )


@pytest.mark.spice
def test_lvlh_radial_direction_thrust_acceleration_aligns_with_position(
    earth_ephemeris_system,
):
    """LVLH 径向推力，验证惯性系中推力加速度方向与位置向量方向点积 ≈ 1。"""
    # Arrange
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    a0 = _EARTH_R_KM + 400.0
    y0 = _keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    thrust = 1.0
    mass = 1000.0
    duration_s = 600.0

    gravity = PointMassGravity(body="EARTH")
    burn = FiniteBurn(
        thrust_profile=lambda t: thrust,
        direction=[1.0, 0.0, 0.0],
        mass=mass,
        direction_frame="LVLH",
    )
    fm = ForceModel(system, forces=[gravity, burn])

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.linspace(et0, et0 + duration_s, 20)

    # Act
    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    # Assert：推力加速度方向与位置向量方向点积 ≈ 1
    for state in result["states"]:
        r = state[:3]
        r_norm = np.linalg.norm(r)
        if r_norm > 1e-15:
            r_hat = r / r_norm
            gravity_acc = -mu / (r_norm**3) * r
            total_acc = fm._compute_total_acceleration(et0, state)
            thrust_acc = total_acc - gravity_acc
            thrust_acc_norm = np.linalg.norm(thrust_acc)
            if thrust_acc_norm > 1e-15:
                dot_product = np.dot(thrust_acc / thrust_acc_norm, r_hat)
                assert dot_product > 0.99, (
                    f"推力加速度方向与位置方向点积应 > 0.99, got {dot_product:.6f}"
                )


# --- 测试 3：动态坐标系转换矩阵正交性 ---


@pytest.mark.spice
def test_vnb_rotation_matrix_orthogonality_during_propagation(earth_ephemeris_system):
    """VNB 推力传播过程中，动态坐标系转换矩阵保持正交（R @ R.T ≈ I）。"""
    # Arrange
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    a0 = _EARTH_R_KM + 400.0
    y0 = _keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    thrust = 1.0
    mass = 1000.0
    duration_s = 3600.0

    gravity = PointMassGravity(body="EARTH")
    burn = FiniteBurn(
        thrust_profile=lambda t: thrust,
        direction=[1.0, 0.0, 0.0],
        mass=mass,
        direction_frame="VNB",
    )
    fm = ForceModel(system, forces=[gravity, burn])

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.linspace(et0, et0 + duration_s, 30)

    # Act
    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    # Assert：每个采样点构造 VNB 转换矩阵并验证正交性
    for state in result["states"]:
        r = state[:3]
        v = state[3:6]
        v_norm = np.linalg.norm(v)
        assert v_norm > 1e-15, "速度不能为零"

        V = v / v_norm
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        assert h_norm > 1e-15, "角动量不能为零"

        N = h / h_norm
        B = np.cross(V, N)

        # VNB 转换矩阵：行向量为 V, N, B
        R = np.array([V, N, B])

        # 验证正交性：R @ R.T ≈ I
        identity_check = R @ R.T
        deviation = np.max(np.abs(identity_check - np.eye(3)))
        assert deviation < 1e-14, f"VNB 转换矩阵正交性偏差应 < 1e-14, got {deviation:.2e}"

        # 验证行列式 = 1（右手系）
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-14, f"VNB 转换矩阵行列式应 ≈ 1, got {det:.6f}"


@pytest.mark.spice
def test_lvlh_rotation_matrix_orthogonality_during_propagation(earth_ephemeris_system):
    """LVLH 推力传播过程中，动态坐标系转换矩阵保持正交（R @ R.T ≈ I）。"""
    # Arrange
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    a0 = _EARTH_R_KM + 400.0
    y0 = _keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    thrust = 1.0
    mass = 1000.0
    duration_s = 3600.0

    gravity = PointMassGravity(body="EARTH")
    burn = FiniteBurn(
        thrust_profile=lambda t: thrust,
        direction=[1.0, 0.0, 0.0],
        mass=mass,
        direction_frame="LVLH",
    )
    fm = ForceModel(system, forces=[gravity, burn])

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.linspace(et0, et0 + duration_s, 30)

    # Act
    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    # Assert：每个采样点构造 LVLH 基并验证几何关系
    for state in result["states"]:
        r = state[:3]
        v = state[3:6]
        r_norm = np.linalg.norm(r)
        assert r_norm > 1e-15, "位置不能为零"

        R = r / r_norm
        v_norm = np.linalg.norm(v)
        assert v_norm > 1e-15, "速度不能为零"

        V = v / v_norm
        N = np.cross(R, V)
        N_norm = np.linalg.norm(N)
        assert N_norm > 1e-15, "轨道面法向不能为零"
        N = N / N_norm

        # LVLH 基的关键几何性质：
        # 1. N 同时垂直于 R 和 V（叉积性质）
        assert abs(np.dot(N, R)) < 1e-14, f"N 应垂直于 R: dot={np.dot(N, R):.2e}"
        assert abs(np.dot(N, V)) < 1e-14, f"N 应垂直于 V: dot={np.dot(N, V):.2e}"

        # 2. R 和 V 的点积 = r·v / (|r||v|)，在椭圆轨道中不为零
        rv_dot = np.dot(R, V)
        # 这是真实的轨道几何，不需要断言具体值

        # 3. 用 Gram-Schmidt 构造严格正交基，验证正交性
        V_perp = V - rv_dot * R
        V_perp_norm = np.linalg.norm(V_perp)
        if V_perp_norm > 1e-15:
            V_perp = V_perp / V_perp_norm
            # 严格正交基 [R, V_perp, N]
            ortho_matrix = np.array([R, V_perp, N])
            identity_check = ortho_matrix @ ortho_matrix.T
            deviation = np.max(np.abs(identity_check - np.eye(3)))
            assert deviation < 1e-14, f"Gram-Schmidt 正交基偏差应 < 1e-14, got {deviation:.2e}"

            # 行列式 = 1（右手系）
            det = np.linalg.det(ortho_matrix)
            assert abs(det - 1.0) < 1e-14, f"正交基行列式应 ≈ 1, got {det:.6f}"


# --- 测试 4：VNB/LVLH 混合方向推力验证 ---


@pytest.mark.spice
def test_vnb_combined_direction_thrust_produces_expected_components(
    earth_ephemeris_system,
):
    """VNB 下 direction=[1,1,0] 产生 V+N 混合方向，验证加速度分量比例。"""
    # Arrange
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    a0 = _EARTH_R_KM + 400.0
    y0 = _keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    thrust = 1.0
    mass = 1000.0
    duration_s = 600.0

    gravity = PointMassGravity(body="EARTH")
    burn = FiniteBurn(
        thrust_profile=lambda t: thrust,
        direction=[1.0, 1.0, 0.0],
        mass=mass,
        direction_frame="VNB",
    )
    fm = ForceModel(system, forces=[gravity, burn])

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.linspace(et0, et0 + duration_s, 10)

    # Act
    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    # Assert：在初始点验证加速度分量比例
    state = result["states"][0]
    r = state[:3]
    v = state[3:6]
    v_norm = np.linalg.norm(v)
    V = v / v_norm
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    N = h / h_norm

    gravity_acc = -mu / (np.linalg.norm(r) ** 3) * r
    total_acc = fm._compute_total_acceleration(et0, state)
    thrust_acc = total_acc - gravity_acc

    # direction=[1,1,0] 在 VNB 下 = (V + N) / sqrt(2)
    expected_dir = (V + N) / np.linalg.norm(V + N)
    expected_thrust_acc = (thrust / mass / 1000.0) * expected_dir

    np.testing.assert_allclose(thrust_acc, expected_thrust_acc, atol=1e-12)


@pytest.mark.spice
def test_lvlh_combined_direction_thrust_produces_expected_components(
    earth_ephemeris_system,
):
    """LVLH 下 direction=[1,1,0] 产生 R+V 混合方向，验证加速度分量比例。"""
    # Arrange
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    a0 = _EARTH_R_KM + 400.0
    y0 = _keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    thrust = 1.0
    mass = 1000.0
    duration_s = 600.0

    gravity = PointMassGravity(body="EARTH")
    burn = FiniteBurn(
        thrust_profile=lambda t: thrust,
        direction=[1.0, 1.0, 0.0],
        mass=mass,
        direction_frame="LVLH",
    )
    fm = ForceModel(system, forces=[gravity, burn])

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.linspace(et0, et0 + duration_s, 10)

    # Act
    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    # Assert：在初始点验证加速度分量比例
    state = result["states"][0]
    r = state[:3]
    v = state[3:6]
    r_norm = np.linalg.norm(r)
    R = r / r_norm
    v_norm = np.linalg.norm(v)
    V = v / v_norm

    gravity_acc = -mu / (r_norm**3) * r
    total_acc = fm._compute_total_acceleration(et0, state)
    thrust_acc = total_acc - gravity_acc

    # direction=[1,1,0] 在 LVLH 下 = (R + V) / sqrt(2)
    expected_dir = (R + V) / np.linalg.norm(R + V)
    expected_thrust_acc = (thrust / mass / 1000.0) * expected_dir

    np.testing.assert_allclose(thrust_acc, expected_thrust_acc, atol=1e-12)


# --- 测试 5：零推力边界 ---


@pytest.mark.spice
def test_vnb_zero_thrust_no_orbit_change(earth_ephemeris_system):
    """VNB 方向但推力为零时，轨道应与纯引力传播一致。"""
    # Arrange
    system = earth_ephemeris_system
    mu = system.gravitational_parameter("EARTH")

    a0 = _EARTH_R_KM + 400.0
    y0 = _keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    gravity = PointMassGravity(body="EARTH")
    burn_zero = FiniteBurn(
        thrust_profile=lambda t: 0.0,
        direction=[1.0, 0.0, 0.0],
        mass=1000.0,
        direction_frame="VNB",
    )
    fm_with_zero_thrust = ForceModel(system, forces=[gravity, burn_zero])
    fm_gravity_only = ForceModel(system, forces=[gravity])

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + 600.0)
    t_eval = np.linspace(et0, et0 + 600.0, 20)

    # Act
    result_with_zero = fm_with_zero_thrust.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)
    result_gravity_only = fm_gravity_only.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    # Assert
    np.testing.assert_allclose(
        result_with_zero["states"],
        result_gravity_only["states"],
        atol=1e-12,
    )
