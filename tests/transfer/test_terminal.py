"""终端条件模块测试。

覆盖 TerminalCondition 抽象基类与 OrbitTerminal、StateTerminal 具体实现。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.transfer.terminal import OrbitTerminal, StateTerminal, TerminalCondition
from e2m2e.data.types.orbit import Orbit

# =============================================================================
# TerminalCondition 抽象基类
# =============================================================================


def test_terminal_condition_is_abstract():
    """TerminalCondition 不能直接实例化。"""
    with pytest.raises(TypeError):
        TerminalCondition()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def earth_moon_system():
    return CR3BP_System(mu=0.012150585, primary="earth", secondary="moon")


@pytest.fixture
def dynamics(earth_moon_system):
    return CR3BP_Dynamics(system=earth_moon_system)


@pytest.fixture
def dummy_orbit(earth_moon_system):
    # DRO 种子初值（Cui et al. 2025，同 tests/algorithms/conftest.py）：稳定周期
    # 轨道，远离月球，CR3BP 积分到任意相位都不发散。测试只用首点与 period，
    # 故 50 点填同值即可。
    state0 = np.array([0.79188556619742, 0.0, 0.0, 0.0, 0.573665890385585, 0.0])
    period = 6.307498
    n = 50
    states = np.tile(state0, (n, 1))
    t = np.linspace(0, period, n)
    orbit = Orbit(states, t, system=earth_moon_system)
    orbit.period = period
    return orbit


@pytest.fixture
def fixed_state():
    return np.array([0.9, 0.1, 0.0, 0.0, 0.5, 0.0])


# =============================================================================
# OrbitTerminal
# =============================================================================


class TestOrbitTerminal:
    def test_get_initial_state_returns_first_state(self, dummy_orbit):
        """get_initial_state 返回轨道首点状态。"""
        terminal = OrbitTerminal(dummy_orbit)
        state = terminal.get_initial_state()
        np.testing.assert_array_equal(state, dummy_orbit.states[0])

    def test_get_initial_state_returns_copy(self, dummy_orbit):
        """get_initial_state 返回副本，不共享内存。"""
        terminal = OrbitTerminal(dummy_orbit)
        state = terminal.get_initial_state()
        state[0] = 999.0
        assert dummy_orbit.states[0, 0] != 999.0

    def test_stores_orbit_reference(self, dummy_orbit):
        """orbit 属性存储轨道引用。"""
        terminal = OrbitTerminal(dummy_orbit)
        assert terminal.orbit is dummy_orbit

    def test_get_arrival_state_propagates(self, dynamics, dummy_orbit):
        """get_arrival_state 通过动力学传播到指定相位。"""
        terminal = OrbitTerminal(dummy_orbit)
        t_ins = dummy_orbit.times[0] + dummy_orbit.period * 0.25
        pos, vel = terminal.get_arrival_state(t_ins, dynamics)

        assert pos.shape == (3,)
        assert vel.shape == (3,)
        assert pos.dtype == np.float64
        assert vel.dtype == np.float64

    def test_get_arrival_state_at_period_boundary(self, dynamics, dummy_orbit):
        """t_ins 在周期边界时返回首点状态（近似）。"""
        terminal = OrbitTerminal(dummy_orbit)
        t0 = float(dummy_orbit.times[0])
        pos, vel = terminal.get_arrival_state(t0, dynamics)

        # t_ins = t0 时，mod 后为 0，应返回首点状态
        np.testing.assert_allclose(pos, dummy_orbit.states[0, :3], atol=1e-12)
        np.testing.assert_allclose(vel, dummy_orbit.states[0, 3:6], atol=1e-12)

    def test_get_arrival_state_at_half_period(self, dynamics, dummy_orbit):
        """t_ins 在半周期时返回与首点不同的状态。"""
        terminal = OrbitTerminal(dummy_orbit)
        t0 = float(dummy_orbit.times[0])
        period = float(dummy_orbit.period)
        t_ins = t0 + period * 0.5
        pos, vel = terminal.get_arrival_state(t_ins, dynamics)

        # 半周期状态应与首点不同
        assert not np.allclose(pos, dummy_orbit.states[0, :3], atol=1e-6)

    def test_get_arrival_state_returns_ndarray(self, dynamics, dummy_orbit):
        """get_arrival_state 返回 numpy 数组。"""
        terminal = OrbitTerminal(dummy_orbit)
        pos, vel = terminal.get_arrival_state(dummy_orbit.times[0], dynamics)
        assert isinstance(pos, np.ndarray)
        assert isinstance(vel, np.ndarray)


# =============================================================================
# StateTerminal
# =============================================================================


class TestStateTerminal:
    def test_get_initial_state_returns_fixed_state(self, fixed_state):
        """get_initial_state 返回固定状态。"""
        terminal = StateTerminal(fixed_state, time=5.0)
        state = terminal.get_initial_state()
        np.testing.assert_array_equal(state, fixed_state)

    def test_get_initial_state_returns_copy(self, fixed_state):
        """get_initial_state 返回副本。"""
        terminal = StateTerminal(fixed_state, time=5.0)
        state = terminal.get_initial_state()
        state[0] = 999.0
        assert terminal.state[0] != 999.0

    def test_get_arrival_state_returns_position_velocity(self, fixed_state, dynamics):
        """get_arrival_state 返回固定状态的位置和速度。"""
        terminal = StateTerminal(fixed_state, time=5.0)
        pos, vel = terminal.get_arrival_state(t_ins=10.0, dynamics=dynamics)

        np.testing.assert_array_equal(pos, fixed_state[:3])
        np.testing.assert_array_equal(vel, fixed_state[3:6])

    def test_get_arrival_state_ignores_t_ins(self, fixed_state, dynamics):
        """get_arrival_state 忽略 t_ins 参数。"""
        terminal = StateTerminal(fixed_state, time=5.0)
        pos1, vel1 = terminal.get_arrival_state(t_ins=0.0, dynamics=dynamics)
        pos2, vel2 = terminal.get_arrival_state(t_ins=100.0, dynamics=dynamics)

        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(vel1, vel2)

    def test_get_arrival_state_ignores_dynamics(self, fixed_state, dynamics):
        """get_arrival_state 忽略 dynamics 参数。"""
        terminal = StateTerminal(fixed_state, time=5.0)
        pos, vel = terminal.get_arrival_state(t_ins=0.0, dynamics=dynamics)
        np.testing.assert_array_equal(pos, fixed_state[:3])
        np.testing.assert_array_equal(vel, fixed_state[3:6])

    def test_time_stored(self, fixed_state):
        """time 属性被正确存储。"""
        terminal = StateTerminal(fixed_state, time=5.0)
        assert terminal.time == pytest.approx(5.0)

    def test_state_converted_to_float(self):
        """state 被转换为 float dtype。"""
        terminal = StateTerminal([1, 2, 3, 4, 5, 6], time=0.0)
        assert terminal.state.dtype == np.float64

    def test_time_converted_to_float(self):
        """time 被转换为 float。"""
        terminal = StateTerminal(np.zeros(6), time=5)
        assert isinstance(terminal.time, float)
        assert terminal.time == pytest.approx(5.0)


# =============================================================================
# 多态接口测试
# =============================================================================


class TestPolymorphism:
    def test_orbit_terminal_is_terminal(self, dummy_orbit):
        """OrbitTerminal 是 TerminalCondition 的子类。"""
        terminal = OrbitTerminal(dummy_orbit)
        assert isinstance(terminal, TerminalCondition)

    def test_state_terminal_is_terminal(self, fixed_state):
        """StateTerminal 是 TerminalCondition 的子类。"""
        terminal = StateTerminal(fixed_state, time=0.0)
        assert isinstance(terminal, TerminalCondition)

    def test_uniform_interface(self, dynamics, dummy_orbit, fixed_state):
        """两种终端条件实现相同的接口。"""
        orbit_term = OrbitTerminal(dummy_orbit)
        state_term = StateTerminal(fixed_state, time=0.0)

        # 两者都有 get_initial_state
        s1 = orbit_term.get_initial_state()
        s2 = state_term.get_initial_state()
        assert s1.shape == (6,)
        assert s2.shape == (6,)

        # 两者都有 get_arrival_state
        p1, v1 = orbit_term.get_arrival_state(dummy_orbit.times[0], dynamics)
        p2, v2 = state_term.get_arrival_state(0.0, dynamics)
        assert p1.shape == (3,)
        assert v1.shape == (3,)
        assert p2.shape == (3,)
        assert v2.shape == (3,)
