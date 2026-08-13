"""Rust 积分器扩展的模块表面与结果对象契约测试。

统一检查公开符号、枚举成员与跨边界结果对象的归一化，
取代过去散落在各方法文件里的 imports smoke 测试。
"""

from types import SimpleNamespace

import pytest

from e2m2e import integrators
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.integrator


def test_module_surface_exports_steppers():
    """公开模块导出全部积分器符号：RK 族、多步族、Cowell、事件路径。"""
    from e2m2e.integrators import (
        MultistepMethod,
        RkMethod,
        cowell_step,
        initialize_abm_history,
        initialize_cowell_history,
        multistep_step,
        rk_step,
        solve_ivp_events,
    )

    assert RkMethod.PD45 is not None
    assert RkMethod.PD78 is not None
    assert RkMethod.RK89 is not None
    assert MultistepMethod.ABM is not None
    for symbol in (
        rk_step,
        multistep_step,
        cowell_step,
        initialize_abm_history,
        initialize_cowell_history,
        solve_ivp_events,
    ):
        assert callable(symbol)


def test_extension_smoke():
    """Rust 扩展模块可导入并响应。"""
    from e2m2e.integrators import hello_integrators

    assert hello_integrators() == "hello from e2m2e-integrators"


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


def test_shooting_result_rejects_inconsistent_status_and_cause():
    raw = SimpleNamespace(
        status="converged",
        cause="max_iterations_reached",
        message="错误组合",
    )

    with pytest.raises(ValueError, match="状态与原因不一致"):
        integrators._ShootingResult(raw)
