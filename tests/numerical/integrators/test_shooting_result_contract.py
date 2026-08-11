"""Rust 打靶结果在 Python 边界的状态契约测试。"""

from types import SimpleNamespace

import pytest

from e2m2e import integrators
from e2m2e.data.templates import ConvergenceState, FailureCause


@pytest.mark.integrator
def test_shooting_result_normalizes_and_validates_contract():
    raw = SimpleNamespace(
        status="converged",
        cause="none",
        message="多重打靶收敛",
        iterations=3,
    )

    result = integrators._ShootingResult(raw)

    assert result.status is ConvergenceState.CONVERGED
    assert result.cause is FailureCause.NONE
    assert result.message == "多重打靶收敛"
    assert result.iterations == 3


@pytest.mark.integrator
def test_shooting_result_rejects_inconsistent_status_and_cause():
    raw = SimpleNamespace(
        status="converged",
        cause="max_iterations_reached",
        message="错误组合",
    )

    with pytest.raises(ValueError, match="状态与原因不一致"):
        integrators._ShootingResult(raw)
