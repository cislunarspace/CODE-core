"""动态坐标系推力传播集成测试（Slice 12 验收，预留 #407）。

验证 VNB 沿速度方向推力提升半长轴、LVLH 径向推力增加偏心率、
转换矩阵正交性与混合方向。

预留状态：``FiniteBurn`` 恒质量低推力从未实现（``to_rust_spec`` 抛
``NotImplementedError``），本文件全部测试标记 ``xfail``；实现后（#407）
去掉 ``pytestmark`` 里的 xfail 即可恢复断言。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.forces import FiniteBurn, ForceModel, PointMassGravity
from tests.numerical.forces.conftest import EARTH_RE, keplerian_to_cartesian, semi_major_axis

pytestmark = [
    pytest.mark.force,
    pytest.mark.low_thrust,
    pytest.mark.xfail(reason="预留 #407：FiniteBurn 恒质量低推力从未实现"),
]


def _propagate_dynamic_axes(
    system,
    direction,
    direction_frame,
    duration_s,
    n_points,
    *,
    e: float = 0.0,
    thrust: float = 1.0,
):
    """构造动态轴推力传播，返回 ``(result, mu, fm, et0)``。

    初值：近圆 LEO（400 km 高度）；推力沿 ``direction``（在
    ``direction_frame`` 下解释）、质量 1000 kg。
    """
    mu = system.gravitational_parameter("EARTH")
    a0 = EARTH_RE + 400.0
    y0 = keplerian_to_cartesian(a0, e, 0.0, 0.0, 0.0, 0.0, mu)

    gravity = PointMassGravity(body="EARTH")
    burn = FiniteBurn(
        thrust_profile=lambda t: thrust,
        direction=direction,
        mass=1000.0,
        direction_frame=direction_frame,
    )
    fm = ForceModel(system, forces=[gravity, burn])

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.linspace(et0, et0 + duration_s, n_points)
    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)
    return result, mu, fm, et0


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


# --- 测试 1：VNB 沿速度方向推力 ---


@pytest.mark.spice
def test_vnb_velocity_direction_thrust_increases_semi_major_axis(earth_icrf_system):
    """VNB 下 direction=[1,0,0] 沿速度方向推力，验证半长轴与轨道能量增加。"""
    result, mu, _, _ = _propagate_dynamic_axes(
        earth_icrf_system, [1.0, 0.0, 0.0], "VNB", 3600.0, 50
    )

    a_initial = semi_major_axis(result["states"][0], mu)
    a_final = semi_major_axis(result["states"][-1], mu)
    assert a_final > a_initial, f"半长轴应增加: initial={a_initial:.3f} km, final={a_final:.3f} km"

    energy_initial = _orbital_energy(result["states"][0], mu)
    energy_final = _orbital_energy(result["states"][-1], mu)
    assert energy_final > energy_initial, (
        f"轨道能量应增加: initial={energy_initial:.6f} km²/s², final={energy_final:.6f} km²/s²"
    )


@pytest.mark.spice
def test_vnb_velocity_direction_thrust_acceleration_aligns_with_velocity(earth_icrf_system):
    """VNB 沿速度方向推力，验证惯性系中推力加速度方向与速度方向点积 ≈ 1。"""
    result, mu, fm, et0 = _propagate_dynamic_axes(
        earth_icrf_system, [1.0, 0.0, 0.0], "VNB", 600.0, 20
    )

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
def test_lvlh_radial_direction_thrust_increases_eccentricity(earth_icrf_system):
    """LVLH 下 direction=[1,0,0] 径向推力，验证偏心率随时间增加。"""
    result, mu, _, _ = _propagate_dynamic_axes(
        earth_icrf_system, [1.0, 0.0, 0.0], "LVLH", 3600.0, 50, e=0.001
    )

    e_initial = _eccentricity(result["states"][0], mu)
    e_final = _eccentricity(result["states"][-1], mu)
    assert e_final > e_initial, f"偏心率应增加: initial={e_initial:.6f}, final={e_final:.6f}"


@pytest.mark.spice
def test_lvlh_radial_direction_thrust_acceleration_aligns_with_position(earth_icrf_system):
    """LVLH 径向推力，验证惯性系中推力加速度方向与位置向量方向点积 ≈ 1。"""
    result, mu, fm, et0 = _propagate_dynamic_axes(
        earth_icrf_system, [1.0, 0.0, 0.0], "LVLH", 600.0, 20
    )

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
                    f"推力加速度方向与位置向量方向点积应 > 0.99, got {dot_product:.6f}"
                )


# --- 测试 3：动态坐标系转换矩阵正交性 ---


@pytest.mark.spice
def test_vnb_rotation_matrix_orthogonality_during_propagation(earth_icrf_system):
    """VNB 推力传播过程中，动态坐标系转换矩阵保持正交（R @ R.T ≈ I）。"""
    result, _, _, _ = _propagate_dynamic_axes(earth_icrf_system, [1.0, 0.0, 0.0], "VNB", 3600.0, 30)

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
def test_lvlh_rotation_matrix_orthogonality_during_propagation(earth_icrf_system):
    """LVLH 推力传播过程中，动态坐标系转换矩阵保持正交（R @ R.T ≈ I）。"""
    result, _, _, _ = _propagate_dynamic_axes(
        earth_icrf_system, [1.0, 0.0, 0.0], "LVLH", 3600.0, 30
    )

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

        # LVLH 基的关键几何性质：N 同时垂直于 R 和 V（叉积性质）
        assert abs(np.dot(N, R)) < 1e-14, f"N 应垂直于 R: dot={np.dot(N, R):.2e}"
        assert abs(np.dot(N, V)) < 1e-14, f"N 应垂直于 V: dot={np.dot(N, V):.2e}"


# --- 测试 4：混合方向 ---


@pytest.mark.spice
def test_vnb_combined_direction_thrust_produces_expected_components(earth_icrf_system):
    """VNB 下 direction=[1,1,1] 混合方向，验证初始点加速度分量比例。"""
    result, mu, fm, et0 = _propagate_dynamic_axes(
        earth_icrf_system, [1.0, 1.0, 1.0], "VNB", 600.0, 10
    )

    state = result["states"][0]
    r = state[:3]
    v = state[3:6]
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    V = v / v_norm
    h = np.cross(r, v)
    N = h / np.linalg.norm(h)
    B = np.cross(V, N)

    gravity_acc = -mu / (r_norm**3) * r
    total_acc = fm._compute_total_acceleration(et0, state)
    thrust_acc = total_acc - gravity_acc

    # direction=[1,1,1] 在 VNB 下 = (V + N + B) / sqrt(3)
    expected_dir = (V + N + B) / np.linalg.norm(V + N + B)
    expected_thrust_acc = (1.0 / 1000.0 / 1000.0) * expected_dir

    np.testing.assert_allclose(thrust_acc, expected_thrust_acc, atol=1e-12)


@pytest.mark.spice
def test_lvlh_combined_direction_thrust_produces_expected_components(earth_icrf_system):
    """LVLH 下 direction=[1,1,0] 混合方向，验证初始点加速度分量比例。"""
    result, mu, fm, et0 = _propagate_dynamic_axes(
        earth_icrf_system, [1.0, 1.0, 0.0], "LVLH", 600.0, 10
    )

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
    expected_thrust_acc = (1.0 / 1000.0 / 1000.0) * expected_dir

    np.testing.assert_allclose(thrust_acc, expected_thrust_acc, atol=1e-12)


# --- 测试 5：零推力边界 ---


@pytest.mark.spice
def test_vnb_zero_thrust_no_orbit_change(earth_icrf_system):
    """VNB 方向但推力为零时，轨道应与纯引力传播一致。"""
    system = earth_icrf_system
    mu = system.gravitational_parameter("EARTH")

    a0 = EARTH_RE + 400.0
    y0 = keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

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

    result_with_zero = fm_with_zero_thrust.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)
    result_gravity_only = fm_gravity_only.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    np.testing.assert_allclose(
        result_with_zero["states"],
        result_gravity_only["states"],
        atol=1e-12,
    )
