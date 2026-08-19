"""DROTRONLPOptimizer 模块测试。

覆盖优化器初始化、变量边界与约束构建。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.transfer import TransferConfig, TransferOptimizationResult
from e2m2e.algorithm.transfer.terminal import OrbitTerminal, StateTerminal
from e2m2e.algorithm.transfer.transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
)
from e2m2e.data.templates import ConvergenceState, FailureCause, TransferType
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


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
    orbit = Orbit(
        states=np.zeros((10, 6)),
        times=np.linspace(0, 10, 10),
        system=earth_moon_system,
    )
    orbit.period = 10.0
    return orbit


@pytest.fixture
def optimizer(dynamics, dummy_orbit):
    departure_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
    return DROTRONLPOptimizer(
        system=dynamics.system,
        dynamics=dynamics,
        departure_orbit=dummy_orbit,
        arrival_orbit=dummy_orbit,
        departure_state=departure_state,
    )


# =============================================================================
# Tests
# =============================================================================


def test_nlp_optimizer_initialization(optimizer):
    assert optimizer is not None
    assert optimizer.mu == 0.012150585
    assert optimizer.departure_state is not None


def test_optimization_variables_roundtrip():
    """NLPOptimizationVariables should round-trip through to_array/from_array."""
    variables = NLPOptimizationVariables(alpha=1.2, transfer_time=15.0, t_ins=3.0)

    arr = variables.to_array()
    assert arr.shape == (3,)
    assert np.allclose(arr, [1.2, 15.0, 3.0])

    rebuilt = NLPOptimizationVariables.from_array(arr)
    assert np.isclose(rebuilt.alpha, variables.alpha)
    assert np.isclose(rebuilt.transfer_time, variables.transfer_time)
    assert np.isclose(rebuilt.t_ins, variables.t_ins)


def test_departure_velocity_direction_preserved(dynamics, dummy_orbit):
    """departure_state 沿任意方向时，出发速度应与原速度方向相同。"""
    departure_state = np.array([0.8, 0.0, 0.0, 0.0, 0.8, 0.0])
    optimizer = DROTRONLPOptimizer(
        system=dynamics.system,
        dynamics=dynamics,
        departure_orbit=dummy_orbit,
        arrival_orbit=dummy_orbit,
        departure_state=departure_state,
    )

    for alpha in [0.8, 1.0, 1.2, 1.5]:
        v_injection = optimizer.compute_departure_velocity(departure_state, alpha)

        original_vel = departure_state[3:]
        original_dir = original_vel / np.linalg.norm(original_vel)
        new_dir = v_injection / np.linalg.norm(v_injection)

        dot = np.dot(original_dir, new_dir)
        # 方向应相同（符号可能相反，取决于轨道类型）
        assert abs(abs(dot) - 1.0) < 1e-6, f"速度方向不一致: dot={dot}"


def test_scipy_optimizer_returns_transfer_result(optimizer):
    """SciPy 路径应直接返回 TransferOptimizationResult。"""
    config = TransferConfig(
        nlp_alpha_min=0.5,
        nlp_alpha_max=2.5,
        nlp_earth_radius=200.0 / 3.84405000e5,
        nlp_moon_radius=100.0 / 3.84405000e5,
        nlp_use_relaxed_velocity=True,
        nlp_velocity_angle_tol=0.05,
    )
    optimizer_with_config = DROTRONLPOptimizer(
        system=optimizer.system,
        dynamics=optimizer.dynamics,
        departure_orbit=optimizer.departure_orbit,
        arrival_orbit=optimizer.arrival_orbit,
        departure_state=optimizer.departure_state,
        config=config,
    )

    result = optimizer_with_config._build_result(
        NLPOptimizationVariables(alpha=1.0, transfer_time=10.0, t_ins=0.0),
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="test",
    )

    assert isinstance(result, TransferOptimizationResult)
    assert result.status is ConvergenceState.CONVERGED
    assert result.cause is FailureCause.NONE
    assert result.total_delta_v >= 0.0
    assert result.transfer_time == pytest.approx(10.0)
    assert result.departure_alpha == pytest.approx(1.0)
    assert result.t_ins == pytest.approx(0.0)


def test_transfer_type_enum_values():
    assert TransferType.DIRECT.value == "direct"
    assert TransferType.LGA.value == "lga"
    assert TransferType.EXTERNAL.value == "external"


def test_module_exports():
    """Top-level transfer 子包应重新导出 NLP 类。"""
    from e2m2e.algorithm.transfer import (
        DROTRONLPOptimizer,
        NLPOptimizationVariables,
        optimize_transfer,
    )

    assert DROTRONLPOptimizer is not None
    assert NLPOptimizationVariables is not None
    assert optimize_transfer is not None


# =============================================================================
# issue #161 — TerminalCondition 接口接入
# =============================================================================


class TestTerminalConditionInterface:
    """验证 ``DROTRONLPOptimizer`` 真正走 ``TerminalCondition`` 接口。"""

    def test_accepts_terminal_pair(self, optimizer):
        """``DROTRONLPOptimizer`` 应通过 ``departure_terminal``/``arrival_terminal`` 接受终端。"""
        opt = DROTRONLPOptimizer(
            system=optimizer.system,
            dynamics=optimizer.dynamics,
            departure_terminal=optimizer.departure_terminal,
            arrival_terminal=optimizer.arrival_terminal,
        )
        assert isinstance(opt.departure_terminal, OrbitTerminal)
        assert isinstance(opt.arrival_terminal, OrbitTerminal)
        # 旧接口仍兼容
        assert opt.departure_orbit is optimizer.departure_orbit
        assert opt.arrival_orbit is optimizer.arrival_orbit

    def test_state_terminal_at_arrival(self, dynamics, dummy_orbit):
        """``StateTerminal`` 应能作为到达端接入（issue #161 验收点）。"""
        fixed_state = np.array([0.95, 0.05, 0.0, 0.0, 0.3, 0.0])
        opt = DROTRONLPOptimizer(
            system=dynamics.system,
            dynamics=dynamics,
            departure_terminal=OrbitTerminal(dummy_orbit),
            arrival_terminal=StateTerminal(fixed_state, time=2.0),
        )
        assert isinstance(opt.arrival_terminal, StateTerminal)
        # StateTerminal 没有 ``orbit`` 属性，``arrival_orbit`` 应为 None
        assert opt.arrival_orbit is None
        # ``get_arrival_state_at_t_ins`` 忽略 ``t_ins``，返回固定状态
        for t_ins in (0.0, 5.0, 100.0):
            pos, vel = opt.get_arrival_state_at_t_ins(t_ins)
            np.testing.assert_array_equal(pos, fixed_state[:3])
            np.testing.assert_array_equal(vel, fixed_state[3:6])

    def test_from_orbits_classmethod(self, dynamics, dummy_orbit):
        """``from_orbits`` 类方法应与旧接口等价（包成 ``OrbitTerminal``）。"""
        departure_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
        opt_from = DROTRONLPOptimizer.from_orbits(
            system=dynamics.system,
            dynamics=dynamics,
            departure_orbit=dummy_orbit,
            arrival_orbit=dummy_orbit,
            departure_state=departure_state,
        )
        opt_legacy = DROTRONLPOptimizer(
            system=dynamics.system,
            dynamics=dynamics,
            departure_orbit=dummy_orbit,
            arrival_orbit=dummy_orbit,
            departure_state=departure_state,
        )
        np.testing.assert_array_equal(opt_from.departure_state, opt_legacy.departure_state)
        assert isinstance(opt_from.departure_terminal, OrbitTerminal)
        assert isinstance(opt_from.arrival_terminal, OrbitTerminal)

    def test_from_orbits_without_state(self, dynamics, dummy_orbit):
        """``from_orbits`` 不显式传 ``departure_state`` 时取轨道首点。"""
        opt = DROTRONLPOptimizer.from_orbits(
            system=dynamics.system,
            dynamics=dynamics,
            departure_orbit=dummy_orbit,
            arrival_orbit=dummy_orbit,
        )
        np.testing.assert_array_equal(opt.departure_state, dummy_orbit.states[0])

    def test_mixing_new_and_legacy_raises(self, dynamics, dummy_orbit):
        """同时传新旧接口参数应报错。"""
        with pytest.raises(ValueError, match="cannot mix"):
            DROTRONLPOptimizer(
                system=dynamics.system,
                dynamics=dynamics,
                departure_terminal=OrbitTerminal(dummy_orbit),
                arrival_terminal=OrbitTerminal(dummy_orbit),
                departure_orbit=dummy_orbit,
            )

    def test_neither_interface_raises(self, dynamics):
        """既不提供终端也不提供 orbit 应报错。"""
        with pytest.raises(ValueError, match="must provide"):
            DROTRONLPOptimizer(
                system=dynamics.system,
                dynamics=dynamics,
            )

    def test_legacy_interface_still_works(self, dynamics, dummy_orbit):
        """旧接口应继续被支持以保持向后兼容。"""
        departure_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
        opt = DROTRONLPOptimizer(
            system=dynamics.system,
            dynamics=dynamics,
            departure_orbit=dummy_orbit,
            arrival_orbit=dummy_orbit,
            departure_state=departure_state,
        )
        assert isinstance(opt.departure_terminal, OrbitTerminal)
        assert isinstance(opt.arrival_terminal, OrbitTerminal)
        np.testing.assert_array_equal(opt.departure_state, departure_state)
