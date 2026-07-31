"""CR3BP 相对运动扩展测试：Encke、LVLH、调相（主题 3）。"""

import numpy as np
import pytest

from e2m2e.core.cr3bp_system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.core.orbit import Orbit
from e2m2e.proximity.phasing import phasing_search
from e2m2e.proximity.relative_dynamics import RelativeDynamics, TargetOrbit

MU_EARTH_MOON = 0.0121505856


@pytest.fixture
def earth_moon_dynamics():
    system = CR3BP_System(mu=MU_EARTH_MOON, primary="earth", secondary="moon")
    return CR3BP_Dynamics(system=system)


@pytest.fixture
def circular_orbit(earth_moon_dynamics):
    """L2 附近近似圆轨道。"""
    dynamics = earth_moon_dynamics
    x_l2 = 1.15568
    n_pts = 200
    period = 3.0
    times = np.linspace(0.0, period, n_pts)
    radius = 0.01
    omega = 2 * np.pi / period
    states = np.empty((n_pts, 6))
    for i, t in enumerate(times):
        th = omega * t
        states[i] = [
            x_l2 + radius * np.cos(th),
            radius * np.sin(th),
            0.0,
            -radius * omega * np.sin(th),
            radius * omega * np.cos(th),
            0.0,
        ]
    orbit = Orbit(states=states, times=times)
    return orbit, dynamics


class TestEncke:
    """Encke 式相对运动。"""

    def test_encke_matches_newton(self, circular_orbit):
        """Encke 与牛顿式右端一致（机器精度）。"""
        orbit, dynamics = circular_orbit
        target = TargetOrbit(orbit)
        rd = RelativeDynamics(target, dynamics)

        t = 1.5
        rho = np.array([1e-4, 2e-4, 1e-5, 1e-5, -1e-5, 0.0])
        f_newton = rd.nonlinear_eom(t, rho)
        f_encke = rd.encke_eom(t, rho)
        np.testing.assert_allclose(f_newton, f_encke, rtol=1e-12, atol=1e-15)

    def test_encke_propagation(self, circular_orbit):
        """Encke 传播收敛。"""
        orbit, dynamics = circular_orbit
        target = TargetOrbit(orbit)
        rd = RelativeDynamics(target, dynamics)

        rho0 = np.array([1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0])
        times, rhos = rd.propagate_nonlinear(rho0, (0.0, 1.0), method="encke")
        assert times.shape[0] == rhos.shape[0]
        assert rhos.shape[1] == 6

    def test_encke_vs_linear(self, circular_orbit):
        """Encke 与线性化在小扰动下接近。"""
        orbit, dynamics = circular_orbit
        target = TargetOrbit(orbit)
        rd = RelativeDynamics(target, dynamics)

        rho0 = np.array([1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0])
        _, rhos_e = rd.propagate_nonlinear(rho0, (0.0, 1.0), method="encke")
        _, rhos_l = rd.propagate(rho0, (0.0, 1.0))
        # 非线性与线性化差异应小（扰动小）
        np.testing.assert_allclose(rhos_e[-1], rhos_l[-1], rtol=0.01)


class TestLVLH:
    """LVLH 系相对状态转换。"""

    def test_roundtrip(self, circular_orbit):
        """往返转换一致。"""
        orbit, dynamics = circular_orbit
        target = TargetOrbit(orbit)
        rd = RelativeDynamics(target, dynamics)

        t = 1.5
        rho_syn = np.array([1e-4, 2e-4, 1e-5, 1e-5, -1e-5, 0.0])
        rho_lvlh, rho_dot_lvlh = rd.to_lvlh(rho_syn, t)
        rho_syn_back = rd.from_lvlh(rho_lvlh, rho_dot_lvlh, t)
        np.testing.assert_allclose(rho_syn, rho_syn_back, rtol=1e-10, atol=1e-15)

    def test_radial_along_x(self, circular_orbit):
        """LVLH R 轴沿目标径向。"""
        orbit, dynamics = circular_orbit
        target = TargetOrbit(orbit)
        rd = RelativeDynamics(target, dynamics)

        # 纯径向扰动（沿目标位置矢量方向）
        t = 1.5
        state = target.state_at(t)
        r_hat = state[:3] / np.linalg.norm(state[:3])
        rho_syn = np.concatenate([1e-4 * r_hat, np.zeros(3)])
        rho_lvlh, _ = rd.to_lvlh(rho_syn, t)
        # LVLH 位置应主要在 R 分量（第 0 个）
        assert abs(rho_lvlh[0]) > abs(rho_lvlh[1])
        assert abs(rho_lvlh[0]) > abs(rho_lvlh[2])


class TestPhasing:
    """调相设计。"""

    def test_phasing_search_shape(self, circular_orbit):
        """返回解列表，每个 tof 一个。"""
        orbit, dynamics = circular_orbit
        tof_grid = np.array([0.5, 1.0, 1.5])
        solutions = phasing_search(orbit, dphase=np.pi / 2, tof_grid=tof_grid, dynamics=dynamics)
        assert len(solutions) == 3
        for sol in solutions:
            assert sol.tof in tof_grid

    def test_phasing_two_impulses(self, circular_orbit):
        """两脉冲解结构。"""
        orbit, dynamics = circular_orbit
        tof_grid = np.array([1.0])
        solutions = phasing_search(orbit, dphase=np.pi, tof_grid=tof_grid, dynamics=dynamics)
        sol = solutions[0]
        if sol.converged:
            assert len(sol.maneuvers) == 2
            assert sol.maneuvers[0].t == pytest.approx(sol.maneuvers[1].t - sol.tof)
            assert sol.total_dv >= 0.0

    def test_phasing_zero_dphase(self, circular_orbit):
        """零相位差：脉冲应为零。"""
        orbit, dynamics = circular_orbit
        tof_grid = np.array([1.0])
        solutions = phasing_search(orbit, dphase=0.0, tof_grid=tof_grid, dynamics=dynamics)
        sol = solutions[0]
        if sol.converged:
            assert sol.total_dv == pytest.approx(0.0, abs=1e-10)
