"""Transfer 类编排与接口测试。

验证 _convert_nlp_result 已移除、optimize 直接按配置分支调度 SciPy / COPT。
"""

from unittest.mock import patch

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.transfer import Transfer, TransferConfig, TransferOptimizationResult
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


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


def test_transfer_dispatches_to_copt_when_enabled(dynamics, dummy_orbit):
    """use_copt 且 COPT 可用时，Transfer 应走 optimize_with_copt 而非 optimizer.optimize。"""
    transfer = Transfer(dynamics).set_orbit(dummy_orbit, dummy_orbit)
    transfer.config.nlp_use_copt = True
    expected_result = TransferOptimizationResult(
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        total_delta_v=1.23,
        transfer_time=12.0,
    )

    with patch("e2m2e.algorithm.transfer.transfer.DROTRONLPOptimizer") as MockOptimizer:
        instance = MockOptimizer.return_value
        instance.optimize.return_value = TransferOptimizationResult(
            status=ConvergenceState.FAILED, cause=FailureCause.UNKNOWN, total_delta_v=99.0
        )
        with (
            patch(
                "e2m2e.algorithm.transfer.transfer.optimize_with_copt",
                return_value=expected_result,
            ) as mock_copt,
            patch("e2m2e.algorithm.transfer.transfer._HAVE_COPT", True),
        ):
            result = transfer.optimize(
                initial_guess={"alpha": 1.0, "transfer_time": 10.0, "t_ins": 5.0},
                alpha_range=(0.5, 2.5),
                t_ins_range=(0.0, 10.0),
            )

    assert result is expected_result
    mock_copt.assert_called_once()
    assert mock_copt.call_args.kwargs["fallback_to_scipy"] is False
    instance.optimize.assert_not_called()


def test_transfer_uses_config_to_initialize_optimizer(dynamics, dummy_orbit):
    """Transfer.optimize 应通过 config 构造优化器，不 poke 属性。"""
    transfer = Transfer(dynamics).set_orbit(dummy_orbit, dummy_orbit)
    expected_result = TransferOptimizationResult(
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        total_delta_v=1.23,
        transfer_time=12.0,
    )

    with patch("e2m2e.algorithm.transfer.transfer.DROTRONLPOptimizer") as MockOptimizer:
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
    assert config.nlp_alpha_range == (0.5, 2.5)
    assert config.nlp_t_ins_range == (0.0, 10.0)
    assert config.nlp_use_relaxed_velocity is True
    assert config.nlp_velocity_angle_tol == pytest.approx(0.05)
    assert config.nlp_verbose is False

    # 确认 optimize() 只被调用一次，且没有通过额外参数覆盖范围
    instance.optimize.assert_called_once()
    call_kwargs = instance.optimize.call_args.kwargs
    assert "alpha_range" not in call_kwargs
    assert "t_ins_range" not in call_kwargs
    assert "use_relaxed_velocity_constraint" not in call_kwargs
    assert "velocity_angle_constraint" not in call_kwargs


def test_transfer_raises_when_copt_unavailable_but_requested(dynamics, dummy_orbit):
    """nlp_use_copt=True 但 coptpy 未安装（_HAVE_COPT=False）时显式报错，
    不静默回退 SciPy（ADR 0020 决策 4）。"""
    transfer = Transfer(dynamics).set_orbit(dummy_orbit, dummy_orbit)
    transfer.config.nlp_use_copt = True

    with (
        patch("e2m2e.algorithm.transfer.transfer.DROTRONLPOptimizer") as MockOptimizer,
        patch("e2m2e.algorithm.transfer.transfer._HAVE_COPT", False),
    ):
        instance = MockOptimizer.return_value
        with pytest.raises(RuntimeError, match="coptpy"):
            transfer.optimize(
                initial_guess={"alpha": 1.0, "transfer_time": 10.0, "t_ins": 5.0},
                alpha_range=(0.5, 2.5),
                t_ins_range=(0.0, 10.0),
            )
    instance.optimize.assert_not_called()
