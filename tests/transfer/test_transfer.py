"""Transfer 类编排与接口测试。

验证 _convert_nlp_result 已移除、optimize 通过 adapter 调用。
"""
from unittest.mock import patch

import numpy as np
import pytest

from e2m2e.core import CR3BP_Dynamics, CR3BP_System
from e2m2e.core.orbit import Orbit
from e2m2e.transfer import Transfer, TransferConfig, TransferOptimizationResult


def test_transfer_has_no_convert_nlp_result_method(dynamics):
    """Transfer 不应再包含 _convert_nlp_result 转换函数。"""
    transfer = Transfer(dynamics)
    assert not hasattr(transfer, "_convert_nlp_result")


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
    """占位轨道 fixture，用于 Transfer 构造。"""
    orbit = Orbit(
        states=np.zeros((10, 6)),
        times=np.linspace(0, 10, 10),
        system=earth_moon_system,
    )
    orbit.period = 10.0
    return orbit


def test_transfer_uses_optimizer_adapter(dynamics, dummy_orbit):
    """Transfer.optimize 应通过 adapter 调用，不再直接调用 optimizer.optimize。"""
    from e2m2e.transfer.optimizers import SciPyTransferOptimizer

    transfer = Transfer(dynamics).set_orbit(dummy_orbit, dummy_orbit)
    expected_result = TransferOptimizationResult(
        success=True,
        total_delta_v=1.23,
        transfer_time=12.0,
    )

    with patch("e2m2e.transfer.transfer.DROTRONLPOptimizer") as MockOptimizer:
        instance = MockOptimizer.return_value
        with patch.object(
            SciPyTransferOptimizer, "optimize", return_value=expected_result
        ) as mock_adapter_optimize:
            result = transfer.optimize(
                initial_guess={"alpha": 1.0, "transfer_time": 10.0, "t_ins": 5.0},
                alpha_range=(0.5, 2.5),
                t_ins_range=(0.0, 10.0),
            )

    assert result is expected_result
    mock_adapter_optimize.assert_called_once()
    instance.optimize.assert_not_called()


def test_transfer_uses_config_to_initialize_optimizer(dynamics, dummy_orbit):
    """Transfer.optimize 应通过 config 构造优化器，不再 poke 属性。"""
    transfer = Transfer(dynamics).set_orbit(dummy_orbit, dummy_orbit)
    expected_result = TransferOptimizationResult(
        success=True,
        total_delta_v=1.23,
        transfer_time=12.0,
    )

    with patch("e2m2e.transfer.transfer.DROTRONLPOptimizer") as MockOptimizer:
        instance = MockOptimizer.return_value
        instance.optimize.return_value = expected_result

        result = transfer.optimize(
            initial_guess={"alpha": 1.0, "transfer_time": 10.0, "t_ins": 5.0},
            alpha_range=(0.5, 2.5),
            t_ins_range=(0.0, 10.0),
            use_relaxed_velocity=True,
            velocity_angle_tol=0.05,
        )

    assert result is expected_result
    MockOptimizer.assert_called_once()
    _, kwargs = MockOptimizer.call_args
    assert "config" in kwargs
    config = kwargs["config"]
    assert isinstance(config, TransferConfig)
    assert config.alpha_range == (0.5, 2.5)
    assert config.t_ins_range == (0.0, 10.0)
    assert config.use_relaxed_velocity is True
    assert config.velocity_angle_tol == pytest.approx(0.05)
    assert config.verbose is False

    # 确认 optimize() 只被调用一次，且没有通过额外参数覆盖范围
    instance.optimize.assert_called_once()
    call_kwargs = instance.optimize.call_args.kwargs
    assert "alpha_range" not in call_kwargs
    assert "t_ins_range" not in call_kwargs
    assert "use_relaxed_velocity_constraint" not in call_kwargs
    assert "velocity_angle_constraint" not in call_kwargs
