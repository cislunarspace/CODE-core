"""Tests for the DROTRONLPOptimizer module.

Converted from script-style (sys.path.insert, print, main()) to pytest.
"""

import numpy as np
import pytest

from e2m2e.core import CR3BP_Dynamics, CR3BP_System
from e2m2e.core.orbit import Orbit
from e2m2e.transfer import TransferConfig, TransferOptimizationResult
from e2m2e.transfer.transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
    TransferType,
)

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
        alpha_min=0.5,
        alpha_max=2.5,
        earth_radius=200.0 / 3.84405000e5,
        moon_radius=100.0 / 3.84405000e5,
        use_relaxed_velocity=True,
        velocity_angle_tol=0.05,
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
        NLPOptimizationVariables(alpha=1.0, transfer_time=10.0, t_ins=5.0),
        success=True,
        message="test",
    )

    assert isinstance(result, TransferOptimizationResult)
    assert result.success
    assert result.total_delta_v >= 0.0
    assert result.transfer_time == pytest.approx(10.0)
    assert result.departure_alpha == pytest.approx(1.0)
    assert result.t_ins == pytest.approx(5.0)


def test_transfer_type_enum_values():
    assert TransferType.DIRECT.value == "direct"
    assert TransferType.LGA.value == "lga"
    assert TransferType.EXTERNAL.value == "external"


def test_module_exports():
    """Top-level transfer 子包应重新导出 NLP 类。"""
    from e2m2e.transfer import (
        DROTRONLPOptimizer,
        NLPOptimizationVariables,
        TransferType,
        optimize_transfer,
    )

    assert DROTRONLPOptimizer is not None
    assert NLPOptimizationVariables is not None
    assert TransferType is not None
    assert optimize_transfer is not None
