"""transfer_optimization 空传播假分类回归测试（#352）。

``_classify_transfer`` 空轨迹不再假装 DIRECT（改 UNKNOWN）；``check_collision``
空轨迹不再假装无碰撞（抛 ``PropagationFailure``，让调用方计为不可行）。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.transfer.transfer_optimization import DROTRONLPOptimizer
from e2m2e.data.templates import TransferType
from e2m2e.data.types.orbit import Orbit
from e2m2e.exceptions import PropagationFailure

pytestmark = pytest.mark.orchestration


@pytest.fixture
def system():
    return CR3BP_System(mu=0.012150585, primary="earth", secondary="moon")


@pytest.fixture
def dynamics(system):
    return CR3BP_Dynamics(system=system)


@pytest.fixture
def dummy_orbit(system):
    orbit = Orbit(
        states=np.zeros((10, 6)),
        times=np.linspace(0, 10, 10),
        system=system,
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


class TestClassifyTransferEmptyTrajectory:
    def test_empty_states_not_direct(self, optimizer):
        """空轨迹返回 UNKNOWN（修复前假装 DIRECT，下游按直达转移处理）。"""
        ttype = optimizer._classify_transfer(
            transfer_time=10.0,
            times=np.empty(0),
            states=np.empty((0, 6)),
            insertion_state=np.zeros(6),
        )
        assert ttype is TransferType.UNKNOWN

    def test_normal_classification_unchanged(self, optimizer):
        """回归：正常轨迹分类不受影响（近地直达仍 DIRECT）。"""
        states = np.zeros((10, 6))
        states[:, 0] = np.linspace(0.5, 1.0, 10)  # x ∈ [0.5, 1.0] < 1.5
        ttype = optimizer._classify_transfer(
            transfer_time=10.0,
            times=np.linspace(0.0, 10.0, 10),
            states=states,
            insertion_state=np.zeros(6),
        )
        assert ttype is TransferType.DIRECT


class TestCheckCollisionEmptyTrajectory:
    def test_empty_states_raises(self, optimizer):
        """空轨迹（传播失败）抛 PropagationFailure，不假装无碰撞（#352）。"""
        y = np.array([1.0, 10.0, 0.0])

        def _empty_integrate(initial_state, t_span):
            return np.array([]), np.array([])

        optimizer.forward_integrate = _empty_integrate  # type: ignore[method-assign]
        with pytest.raises(PropagationFailure, match="轨迹为空"):
            optimizer.check_collision(y)
