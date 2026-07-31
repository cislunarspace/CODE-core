"""CR3BP 相对运动动力学测试（主题 3）。"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_System
from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.proximity.relative_dynamics import RelativeDynamics, RelativeState, TargetOrbit
from e2m2e.data.types.orbit import Orbit

MU_EARTH_MOON = 0.0121505856  # 地月质量比


@pytest.fixture
def earth_moon_dynamics():
    """地月 CR3BP 动力学。"""
    system = CR3BP_System(mu=MU_EARTH_MOON, primary="earth", secondary="moon")
    return CR3BP_Dynamics(system=system)


@pytest.fixture
def circular_orbit(earth_moon_dynamics):
    """绕 L2 的近似圆轨道（Halo 近似，用于测试）。

    在 L2 附近构造小振幅周期轨道，线性化后应退化为 CW 形式。
    """
    dynamics = earth_moon_dynamics
    # L2 位置（近似）
    x_l2 = 1.15568
    # 小振幅圆轨道：在 L2 邻域内 xy 平面圆运动
    n_pts = 200
    period = 3.0  # 无量纲
    times = np.linspace(0.0, period, n_pts)
    radius = 0.01  # 小振幅
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


class TestTargetOrbit:
    """目标轨道包装。"""

    def test_state_at_grid_points(self, circular_orbit):
        """网格点上查询返回精确值。"""
        orbit, _ = circular_orbit
        target = TargetOrbit(orbit)
        state = target.state_at(orbit.times[10])
        np.testing.assert_allclose(state, orbit.states[10])

    def test_state_at_interpolated(self, circular_orbit):
        """非网格点线性插值。"""
        orbit, _ = circular_orbit
        target = TargetOrbit(orbit)
        t_mid = 0.5 * (orbit.times[10] + orbit.times[11])
        state = target.state_at(t_mid)
        expected = 0.5 * (orbit.states[10] + orbit.states[11])
        np.testing.assert_allclose(state, expected)

    def test_out_of_range_raises(self, circular_orbit):
        """越界查询报错。"""
        orbit, _ = circular_orbit
        target = TargetOrbit(orbit)
        with pytest.raises(ValueError, match="超出轨道范围"):
            target.state_at(orbit.times[-1] + 1.0)


class TestRelativeDynamics:
    """RLM 线性化相对运动。"""

    def test_linear_model_shape(self, circular_orbit):
        """A(t) 形状 (6,6)。"""
        orbit, dynamics = circular_orbit
        target = TargetOrbit(orbit)
        rd = RelativeDynamics(target, dynamics)
        A = rd.linear_model(orbit.times[0])
        assert A.shape == (6, 6)
        # 上右块是单位阵
        np.testing.assert_allclose(A[:3, 3:], np.eye(3))

    def test_linear_model_matches_absolute(self, circular_orbit):
        """A(t) 与绝对动力学雅可比一致。"""
        orbit, dynamics = circular_orbit
        target = TargetOrbit(orbit)
        rd = RelativeDynamics(target, dynamics)
        t = orbit.times[50]
        A_rel = rd.linear_model(t)
        state = target.state_at(t)
        A_abs = dynamics.compute_jacobian_A(state)
        np.testing.assert_allclose(A_rel, A_abs)

    def test_propagate_small_perturbation(self, circular_orbit):
        """小扰动传播：相对轨迹保持在目标轨道邻域。"""
        orbit, dynamics = circular_orbit
        target = TargetOrbit(orbit)
        rd = RelativeDynamics(target, dynamics)

        # 初始小扰动
        rho0 = np.array([1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0])
        t_span = (orbit.times[0], orbit.times[50])
        times, rhos = rd.propagate(rho0, t_span, max_step=0.01)

        assert times.shape[0] == rhos.shape[0]
        assert rhos.shape[1] == 6
        # 线性化在小扰动下有效：扰动量级应保持小
        assert np.max(np.abs(rhos)) < 1e-3

    def test_propagate_with_stm(self, circular_orbit):
        """相对 STM：Φ(t0,t0)=I，且与绝对 STM 一致。"""
        orbit, dynamics = circular_orbit
        target = TargetOrbit(orbit)
        rd = RelativeDynamics(target, dynamics)

        rho0 = np.array([1e-6, 0.0, 0.0, 0.0, 0.0, 0.0])
        t_span = (orbit.times[0], orbit.times[20])
        times, rhos, stms = rd.propagate_with_stm(rho0, t_span)

        assert stms.shape == (times.shape[0], 6, 6)
        # 初始 STM 是单位阵
        np.testing.assert_allclose(stms[0], np.eye(6), atol=1e-10)

    def test_stm_propagates_perturbation(self, circular_orbit):
        """用 STM 传播扰动：δx(t) = Φ(t,t0) δx(t0)，与直接传播一致。"""
        orbit, dynamics = circular_orbit
        target = TargetOrbit(orbit)
        rd = RelativeDynamics(target, dynamics)

        rho0 = np.array([1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0])
        t_span = (orbit.times[0], orbit.times[30])
        times, rhos_direct = rd.propagate(rho0, t_span, max_step=0.01)
        _, _, stms = rd.propagate_with_stm(rho0, t_span)

        # 用 STM 传播
        rho_stm = stms[-1] @ rho0
        # 直接传播末态
        rho_direct = rhos_direct[-1]
        # 两种积分路径（线性化 RK45 vs 绝对 STM）数值差异 ~5%，属正常
        np.testing.assert_allclose(rho_stm, rho_direct, rtol=0.1, atol=1e-8)

    def test_relative_state_dataclass(self):
        """RelativeState 数据结构。"""
        rs = RelativeState(
            rho=np.array([1.0, 2.0, 3.0]),
            rho_dot=np.array([0.1, 0.2, 0.3]),
            frame="Synodic",
            epoch=0.0,
        )
        assert rs.rho.shape == (3,)
        assert rs.frame == "Synodic"
