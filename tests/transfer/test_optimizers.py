"""Transfer 优化器适配与导出测试。

验证 optimizer adapter 调用与 NLPOptimizationResult 不再公开导出。
"""

import numpy as np
import pytest

from e2m2e.core import CR3BP_Dynamics, CR3BP_System
from e2m2e.core.orbit import Orbit
from e2m2e.transfer import TransferOptimizationResult
from e2m2e.transfer.transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
)


def test_module_does_not_export_nlp_result():
    """NLPOptimizationResult 不应再从 e2m2e.transfer 公共导出。"""
    import e2m2e.transfer

    assert not hasattr(e2m2e.transfer, "NLPOptimizationResult")


@pytest.fixture
def earth_moon_system():
    """地月 CR3BP 系统 fixture。"""
    return CR3BP_System(mu=0.012150585, primary="earth", secondary="moon")


@pytest.fixture
def dynamics(earth_moon_system):
    """CR3BP 动力学 fixture。"""
    return CR3BP_Dynamics(system=earth_moon_system)


@pytest.fixture
def dummy_orbit(earth_moon_system):
    """占位轨道 fixture，用于优化器构造。"""
    orbit = Orbit(
        states=np.zeros((10, 6)),
        times=np.linspace(0, 10, 10),
        system=earth_moon_system,
    )
    orbit.period = 10.0
    return orbit


@pytest.fixture
def optimizer(dynamics, dummy_orbit):
    """预配置的 DROTRONLPOptimizer fixture。"""
    departure_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
    return DROTRONLPOptimizer(
        system=dynamics.system,
        dynamics=dynamics,
        departure_orbit=dummy_orbit,
        arrival_orbit=dummy_orbit,
        departure_state=departure_state,
    )


def test_scipy_adapter_returns_transfer_result(optimizer):
    """SciPy adapter 应统一返回 TransferOptimizationResult。"""
    from e2m2e.transfer.optimizers import SciPyTransferOptimizer

    expected = TransferOptimizationResult(success=True, total_delta_v=1.23)
    optimizer.optimize = lambda **kwargs: expected

    adapter = SciPyTransferOptimizer(optimizer)
    initial_guess = NLPOptimizationVariables(alpha=1.0, transfer_time=10.0, t_ins=5.0)

    result = adapter.optimize(initial_guess)

    assert result is expected
    assert isinstance(result, TransferOptimizationResult)
