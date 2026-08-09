"""HMN 霍曼直接转移测试（ADR 0013 物理定义验证）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from e2m2e.algorithm.transfer import (
    HmnTransferDetails,
    TransferDesignResult,
    transfer_orbit,
)
from e2m2e.algorithm.transfer.hohmann import (
    MU_EARTH,
    R_EARTH,
    TliParams,
    _rotation_matrix,
    construct_departure_state,
    ephemeris_shoot_transfer,
    hohmann_delta_v,
    hohmann_tof,
    keplerian_to_cartesian,
    scan_lambert_delta_v,
)


class TestRotationMatrix:
    """旋转矩阵单元测试。"""

    def test_identity_when_all_zero(self):
        """i=0, ω=0, Ω=0 时 R = I。"""
        r = _rotation_matrix(0.0, 0.0, 0.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-14)

    def test_orthogonal(self):
        """R 为正交矩阵：R^T R = I, det(R) = 1。"""
        r = _rotation_matrix(30.0, 45.0, 60.0)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)
        assert abs(np.linalg.det(r) - 1.0) < 1e-14

    def test_90deg_inclination(self):
        """i=90° 时 z 轴翻转。"""
        r = _rotation_matrix(90.0, 0.0, 0.0)
        # R₁(-90°): [1,0,0; 0,0,1; 0,-1,0] 的转置形式
        expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        np.testing.assert_allclose(r, expected, atol=1e-14)


class TestKeplerianToCartesian:
    """开普勒根数→笛卡尔状态 单元测试。"""

    def test_circular_orbit_speed(self):
        """圆轨道速度 = √(μ/r)，误差 < 0.01%。"""
        r_park = R_EARTH + 200.0
        r_eci, v_eci = keplerian_to_cartesian(
            a_km=r_park,
            e=0.0,
            i_deg=0.0,
            omega_deg=0.0,
            raan_deg=0.0,
            nu_deg=0.0,
        )
        expected_v = math.sqrt(MU_EARTH / r_park)
        v_mag = np.linalg.norm(v_eci)
        assert abs(v_mag - expected_v) / expected_v < 1e-6

    def test_position_magnitude(self):
        """位置矢量模 = r (轨道半径)。"""
        a = 42164.0  # GEO
        e = 0.0
        r_eci, _ = keplerian_to_cartesian(a, e, 0.0, 0.0, 0.0, 0.0)
        assert abs(np.linalg.norm(r_eci) - a) < 1e-6

    def test_elliptic_apogee(self):
        """椭圆轨道远地点：r = a(1+e)。"""
        a = 20000.0
        e = 0.5
        r_eci, _ = keplerian_to_cartesian(a, e, 0.0, 0.0, 0.0, 180.0)
        expected_r = a * (1.0 + e)
        assert abs(np.linalg.norm(r_eci) - expected_r) < 1e-6

    def test_inclination_effect(self):
        """i=90° 时，速度 z 分量最大；i=0° 时 z 分量为 0。"""
        r_park = R_EARTH + 200.0
        _, v0 = keplerian_to_cartesian(r_park, 0.0, 0.0, 0.0, 0.0, 0.0)
        _, v90 = keplerian_to_cartesian(r_park, 0.0, 90.0, 0.0, 0.0, 0.0)
        assert abs(v0[2]) < 1e-10
        assert abs(v90[2]) > 1e-3  # z 分量应显著非零


class TestConstructDepartureState:
    """TLI 出发状态构造测试。"""

    def test_zero_inclination_circular(self):
        """i=0, γ=0 时：r 在 xy 平面，v 沿 y 方向。"""
        params = TliParams(parking_alt_km=200.0, inclination_deg=0.0)
        r, v = construct_departure_state(params)
        r_park = R_EARTH + 200.0
        expected_v = math.sqrt(MU_EARTH / r_park)
        assert abs(np.linalg.norm(r) - r_park) < 1e-6
        assert abs(np.linalg.norm(v) - expected_v) / expected_v < 1e-6
        assert abs(r[2]) < 1e-10  # z = 0
        assert abs(v[0]) < 1e-10  # vx ≈ 0

    def test_inclination_90_deg(self):
        """i=90° 时，速度有 z 分量。"""
        params = TliParams(parking_alt_km=200.0, inclination_deg=90.0)
        r, v = construct_departure_state(params)
        assert abs(v[2]) > 1e-3  # vz 应显著非零

    def test_speed_preserved_after_rotation(self):
        """旋转后速度大小不变。"""
        params = TliParams(parking_alt_km=200.0, inclination_deg=45.0, raan_deg=30.0)
        r, v = construct_departure_state(params)
        r_park = R_EARTH + 200.0
        expected_v = math.sqrt(MU_EARTH / r_park)
        assert abs(np.linalg.norm(v) - expected_v) / expected_v < 1e-10

    def test_nonzero_flight_path_angle_raises(self):
        """γ≠0 时应抛出 NotImplementedError。"""
        params = TliParams(parking_alt_km=200.0, inclination_deg=0.0, flight_path_angle_deg=5.0)
        with pytest.raises(NotImplementedError, match="非零航迹角"):
            construct_departure_state(params)


class TestHohmannDeltaV:
    """霍曼转移 Δv 验证。"""

    def test_leo_to_geo(self):
        """LEO 300km -> GEO: Delta-v via Hohmann formula (Curtis Sec.6.1)."""
        r1 = R_EARTH + 300.0  # 6678.137 km
        r2 = 42164.0  # GEO
        dv1, dv2 = hohmann_delta_v(r1, r2)
        # dv1 = sqrt(mu/r1)*(sqrt(2r2/(r1+r2)) - 1) ~ 2.426 km/s
        # dv2 = sqrt(mu/r2)*(1 - sqrt(2r1/(r1+r2))) ~ 1.467 km/s
        assert abs(dv1 - 2.426) < 0.01
        assert abs(dv2 - 1.467) < 0.01

    def test_leo_to_moon(self):
        """LEO 200km → 月球轨道：Δv ~ 3.13 / 0.83 km/s。"""
        r1 = R_EARTH + 200.0
        r2 = 384405.0  # 地月平均距离
        dv1, dv2 = hohmann_delta_v(r1, r2)
        assert abs(dv1 - 3.13) < 0.05
        assert abs(dv2 - 0.83) < 0.05

    def test_symmetric_circular(self):
        """相同半径的圆轨道间：Δv = 0。"""
        dv1, dv2 = hohmann_delta_v(42164.0, 42164.0)
        assert abs(dv1) < 1e-10
        assert abs(dv2) < 1e-10


class TestHohmannTof:
    """霍曼转移飞行时间验证。"""

    def test_leo_to_moon_tof(self):
        """LEO 200km -> Moon (384405 km): TOF ~ 4.98 days."""
        r1 = R_EARTH + 200.0
        r2 = 384405.0
        tof = hohmann_tof(r1, r2)
        tof_days = tof / 86400.0
        assert abs(tof_days - 4.98) < 0.1

    def test_leo_to_geo_tof(self):
        """LEO 300km → GEO：TOF ≈ 5.26 小时。"""
        r1 = R_EARTH + 300.0
        r2 = 42164.0
        tof = hohmann_tof(r1, r2)
        tof_hours = tof / 3600.0
        assert abs(tof_hours - 5.26) < 0.1

    def test_half_ellipse_period(self):
        """TOF = π·√(a_t³/μ)，其中 a_t = (r1+r2)/2。"""
        r1 = R_EARTH + 200.0
        r2 = 384405.0
        a_t = (r1 + r2) / 2.0
        expected = math.pi * math.sqrt(a_t**3 / MU_EARTH)
        assert abs(hohmann_tof(r1, r2) - expected) < 1e-6


class TestTransferOrbitHmn:
    """transfer_orbit("HMN") 端到端测试。"""

    def test_returns_transfer_design_result(self):
        """transfer_orbit("HMN") 返回 TransferDesignResult，transfer_type == "HMN"。"""
        params = TliParams(parking_alt_km=200.0, inclination_deg=0.0)
        result = transfer_orbit("HMN", tli_params=params, target_orbit_radius_km=384405.0)
        assert isinstance(result, TransferDesignResult)
        assert result.transfer_type == "HMN"

    def test_delta_v_matches_analytical(self):
        """result.delta_v 与 hohmann_delta_v(r1, r2) 之和一致。"""
        params = TliParams(parking_alt_km=200.0, inclination_deg=0.0)
        r2 = 384405.0
        result = transfer_orbit("HMN", tli_params=params, target_orbit_radius_km=r2)
        r1 = R_EARTH + 200.0
        dv1, dv2 = hohmann_delta_v(r1, r2)
        assert abs(result.delta_v - (dv1 + dv2)) < 1e-10

    def test_tof_in_details(self):
        """details 中 tof_sec 与 hohmann_tof(r1, r2) 一致。"""
        params = TliParams(parking_alt_km=200.0, inclination_deg=0.0)
        r2 = 384405.0
        result = transfer_orbit("HMN", tli_params=params, target_orbit_radius_km=r2)
        r1 = R_EARTH + 200.0
        expected_tof = hohmann_tof(r1, r2)
        assert isinstance(result.details, HmnTransferDetails)
        assert abs(result.details.tof_sec - expected_tof) < 1e-10

    def test_unsupported_type_raises(self):
        """transfer_orbit("low_thrust") without engine_config 抛出 ValueError。"""
        with pytest.raises(ValueError, match="low_thrust"):
            transfer_orbit("low_thrust")


class TestHmnTransferPhysics:
    """霍曼转移物理守恒量验证。"""

    # LEO 300km → GEO 典型场景
    _r1 = R_EARTH + 300.0
    _r2 = 42164.0

    def test_transfer_semi_major_axis(self):
        """转移轨道半长轴 a_t = (r1 + r2) / 2。"""
        r1, r2 = self._r1, self._r2
        a_t_expected = (r1 + r2) / 2.0
        dv1, _ = hohmann_delta_v(r1, r2)
        v_circ = math.sqrt(MU_EARTH / r1)
        v1 = v_circ + dv1  # 出发点速度（切向加速后）
        # 从 vis-viva 反算半长轴: 1/a = 2/r - v²/μ
        a_t_actual = 1.0 / (2.0 / r1 - v1**2 / MU_EARTH)
        assert abs(a_t_actual - a_t_expected) / a_t_expected < 0.001

    def test_energy_conservation_on_transfer_ellipse(self):
        """转移椭圆上出发点和到达点比机械能相等。"""
        r1, r2 = self._r1, self._r2
        dv1, _ = hohmann_delta_v(r1, r2)
        v1 = math.sqrt(MU_EARTH / r1) + dv1  # 出发点速度
        # 活力公式求到达点速度: v² = 2μ/r2 + v1² - 2μ/r1
        v2 = math.sqrt(2.0 * MU_EARTH / r2 + v1**2 - 2.0 * MU_EARTH / r1)
        eps1 = v1**2 / 2.0 - MU_EARTH / r1
        eps2 = v2**2 / 2.0 - MU_EARTH / r2
        assert abs(eps1 - eps2) < 1e-6

    def test_angular_momentum_conservation(self):
        """共面霍曼转移比角动量守恒。"""
        r1, r2 = self._r1, self._r2
        dv1, _ = hohmann_delta_v(r1, r2)
        v1 = math.sqrt(MU_EARTH / r1) + dv1  # 出发点切向速度
        v2 = math.sqrt(2.0 * MU_EARTH / r2 + v1**2 - 2.0 * MU_EARTH / r1)
        # 切向速度: |r×v| = r * v
        h1 = r1 * v1
        h2 = r2 * v2
        assert abs(h1 - h2) / h1 < 1e-10

    def test_arrival_velocity_from_vis_viva(self):
        """活力公式到达速度与 hohmann_delta_v dv2 一致。"""
        r1, r2 = self._r1, self._r2
        _, dv2 = hohmann_delta_v(r1, r2)
        a_t = (r1 + r2) / 2.0
        # vis-viva: v = sqrt(μ * (2/r - 1/a))
        v2_vis_viva = math.sqrt(MU_EARTH * (2.0 / r2 - 1.0 / a_t))
        # 与圆轨道速度 - dv2 比较: dv2 = v_circ2 - v2 => v2 = v_circ2 - dv2
        v_circ2 = math.sqrt(MU_EARTH / r2)
        v2_from_dv = v_circ2 - dv2
        assert abs(v2_vis_viva - v2_from_dv) < 1e-10


class TestLambertBatchScan:
    """Lambert 批量扫描 Δv 测试。"""

    def test_scan_finds_hohmann_tof(self):
        """对 LEO→月球场景，扫描 [3, 7] 天的 tof 网格，最优 tof 接近 hohmann_tof。"""
        r1 = R_EARTH + 200.0
        r2 = 384405.0  # 地月平均距离
        expected_tof = hohmann_tof(r1, r2)

        # 出发状态：停泊轨道（x 轴方向，圆轨道）
        r0 = np.array([r1, 0.0, 0.0])
        v0_park = np.array([0.0, math.sqrt(MU_EARTH / r1), 0.0])

        # 目标位置：负 x 轴（180° 转移角，与霍曼转移几何一致）
        r_target = np.array([-r2, 0.0, 0.0])
        # 目标速度：圆轨道近似
        v_target = np.array([0.0, -math.sqrt(MU_EARTH / r2), 0.0])

        # 扫描 [3, 7] 天的 tof 网格
        tof_grid = np.linspace(3.0 * 86400.0, 7.0 * 86400.0, 50)

        optimal_tof, _, _ = scan_lambert_delta_v(r0, v0_park, r_target, v_target, tof_grid)

        optimal_tof_days = optimal_tof / 86400.0
        expected_tof_days = expected_tof / 86400.0
        assert abs(optimal_tof_days - expected_tof_days) < 0.5

    def test_scan_dv_less_than_fixed(self):
        """扫描的 Δv ≤ 固定 tof 的 Δv（扫描不会更差）。"""
        r1 = R_EARTH + 200.0
        r2 = 384405.0

        r0 = np.array([r1, 0.0, 0.0])
        v0_park = np.array([0.0, math.sqrt(MU_EARTH / r1), 0.0])
        r_target = np.array([-r2, 0.0, 0.0])
        v_target = np.array([0.0, -math.sqrt(MU_EARTH / r2), 0.0])

        # 固定 tof = hohmann_tof
        fixed_tof = hohmann_tof(r1, r2)
        fixed_result = scan_lambert_delta_v(r0, v0_park, r_target, v_target, np.array([fixed_tof]))
        fixed_dv = np.linalg.norm(fixed_result[1] - v0_park) + np.linalg.norm(
            fixed_result[2] - v_target
        )

        # 扫描 [3, 7] 天
        tof_grid = np.linspace(3.0 * 86400.0, 7.0 * 86400.0, 50)
        optimal_tof, v0_opt, vf_opt = scan_lambert_delta_v(
            r0, v0_park, r_target, v_target, tof_grid
        )
        scan_dv = np.linalg.norm(v0_opt - v0_park) + np.linalg.norm(vf_opt - v_target)

        assert scan_dv <= fixed_dv + 0.01  # 允许网格离散化误差


# ---------------------------------------------------------------------------
# TwoBodyDynamics：纯 Python 二体动力学（测试用，不依赖 SPICE）
# ---------------------------------------------------------------------------


class TwoBodyDynamics:
    """简化的二体动力学（测试用），满足 MultipleShooting 接口。

    实现 RK4 积分 + 42 维增广状态（含 STM 变分方程）。
    """

    def __init__(self, mu: float = MU_EARTH):
        self.mu = mu

    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        """二体运动方程：r' = v, v' = -mu*r/|r|^3。"""
        r = state[:3]
        v = state[3:]
        r_norm = np.linalg.norm(r)
        a = -self.mu * r / r_norm**3
        return np.concatenate([v, a])

    def _stm_eom(self, t: float, aug_state: np.ndarray) -> np.ndarray:
        """42 维增广状态运动方程（含 STM 变分方程）。"""
        r = aug_state[:3]
        v = aug_state[3:6]
        r_norm = np.linalg.norm(r)
        r3 = r_norm**3
        r5 = r_norm**5

        # 状态导数
        dr = v
        dv = -self.mu * r / r3

        # 状态转移矩阵导数: dPhi = A * Phi
        # A = [[0, I], [dg/dr, 0]]
        dg = self.mu * (3.0 * np.outer(r, r) / r5 - np.eye(3) / r3)
        stm = aug_state[6:].reshape(6, 6)
        dstm = np.zeros_like(stm)
        dstm[:3, :] = stm[3:6, :]  # dPhi_r = Phi_v
        dstm[3:6, :] = dg @ stm[:3, :]  # dPhi_v = dg * Phi_r

        return np.concatenate([dr, dv, dstm.ravel()])

    def propagate(
        self,
        state: np.ndarray,
        t_span: tuple[float, float],
        with_stm: bool = True,
    ) -> dict:
        """RK4 积分传播（含 STM）。"""
        state = np.asarray(state, dtype=float)
        if with_stm:
            stm0 = np.eye(6).ravel()
            aug0 = np.concatenate([state, stm0])
            n_total = 42
        else:
            aug0 = state.copy()
            n_total = 6

        # RK4 积分参数
        t0, tf = t_span
        n_steps = 200
        dt = (tf - t0) / n_steps

        times = np.empty(n_steps + 1)
        states = np.empty((n_steps + 1, n_total))
        times[0] = t0
        states[0] = aug0.copy()

        y = aug0.copy()
        t_cur = t0
        eom = self._stm_eom if with_stm else self.equations_of_motion

        for i in range(n_steps):
            k1 = eom(t_cur, y)
            k2 = eom(t_cur + dt / 2, y + dt / 2 * k1)
            k3 = eom(t_cur + dt / 2, y + dt / 2 * k2)
            k4 = eom(t_cur + dt, y + dt * k3)
            y = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            t_cur += dt
            times[i + 1] = t_cur
            states[i + 1] = y.copy()

        result: dict = {
            "time": times,
            "states": states[:, :6],
        }
        if with_stm:
            result["stm"] = states[:, 6:].reshape(-1, 6, 6)
        return result


class TestEphemerisShooting:
    """ephemeris_shoot_transfer 打靶收敛测试（二体动力学，纯 Python）。"""

    def _make_lambert_guess(
        self,
        r1_km: float,
        r2_km: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """构造 LEO→远轨道的 Lambert 初猜（共面、沿 y 轴出发）。"""
        r0 = np.array([r1_km, 0.0, 0.0])
        v_circ = math.sqrt(MU_EARTH / r1_km)
        dv1, _ = hohmann_delta_v(r1_km, r2_km)
        v0 = np.array([0.0, v_circ + dv1, 0.0])
        tof = hohmann_tof(r1_km, r2_km)
        return r0, v0, tof

    def test_shooting_converges_two_body(self):
        """TwoBodyDynamics + Lambert 初猜，MultipleShooting 应收敛。"""
        r1 = R_EARTH + 300.0
        r2 = R_EARTH + 35786.0  # GEO 转移
        r0, v0, tof = self._make_lambert_guess(r1, r2)

        dyn = TwoBodyDynamics(mu=MU_EARTH)
        result = ephemeris_shoot_transfer(
            dynamics=dyn,
            t0=0.0,
            r0=r0,
            v0=v0,
            tof=tof,
            n_patches=5,
            max_iter=30,
            tolerance=1e-6,
        )

        assert result.converged is True
        assert result.max_residual < 1e-6
        assert result.outer_iterations <= 30

    def test_shooting_preserves_trajectory_shape(self):
        """打靶后各 patch point 位置模在合理范围内（r1 到 r2 之间）。"""
        r1 = R_EARTH + 200.0
        r2 = R_EARTH + 35786.0  # GEO
        r0, v0, tof = self._make_lambert_guess(r1, r2)

        dyn = TwoBodyDynamics(mu=MU_EARTH)
        result = ephemeris_shoot_transfer(
            dynamics=dyn,
            t0=0.0,
            r0=r0,
            v0=v0,
            tof=tof,
            n_patches=5,
            max_iter=30,
            tolerance=1e-6,
        )

        assert result.converged is True
        radii = np.linalg.norm(result.state_patch[:, :3], axis=1)
        # 出发点半径应在 LEO 附近
        assert radii[0] < r1 * 1.5
        # 所有半径应为正（不应穿入地球中心）
        assert np.all(radii > 0)

    def test_shooting_insufficient_patches_raises(self):
        """n_patches < 2 时应抛出 ValueError。"""
        dyn = TwoBodyDynamics(mu=MU_EARTH)
        r0 = np.array([R_EARTH + 300.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.7, 0.0])
        with pytest.raises(ValueError, match="n_patches must be >= 2"):
            ephemeris_shoot_transfer(
                dynamics=dyn,
                t0=0.0,
                r0=r0,
                v0=v0,
                tof=3600.0,
                n_patches=1,
            )

    def test_transfer_orbit_hmn_with_dynamics(self):
        """transfer_orbit('HMN', dynamics=...) 应返回含 trajectory 的结果。"""
        dyn = TwoBodyDynamics(mu=MU_EARTH)
        params = TliParams(parking_alt_km=300.0, inclination_deg=0.0)
        result = transfer_orbit(
            "HMN",
            tli_params=params,
            target_orbit_radius_km=R_EARTH + 35786.0,
            dynamics=dyn,
        )
        assert isinstance(result, TransferDesignResult)
        assert result.transfer_type == "HMN"
        # 打靶成功时 trajectory 应为 (N, 6) 数组
        assert result.trajectory is not None
        assert result.trajectory.ndim == 2
        assert result.trajectory.shape[1] == 6
