"""``PropagationFailure`` 类型异常测试（ADR 0020 迁移第 1 步）。

第 1 步是地基：引入 ``PropagationFailure(E2M2EError)``，在 Rust→Python FFI
边界把 CR3BP 步长塌缩从"靠错误消息字符串前缀匹配"升级为"按异常类型捕获"。
本步零行为变更——步塌缩时 ``_propagate_state_only`` 仍返回空 states。

验证三点：
1. ``PropagationFailure`` 是 ``E2M2EError`` 子类（统一捕获契约），
   且不是 ``RuntimeError`` 子类（与通用运行时错误区分）。
2. CR3BP 步长塌缩经 FFI 表现为 ``PropagationFailure``（仓内 Rust 测试
   ``step_collapse_early_exit`` 的 Python 侧对应）。
3. ``_propagate_state_only`` 步塌缩仍返回空 states（下游 NLP ``len==0``
   嗅探依赖，回归保护）。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.data.constants import Datum
from e2m2e.exceptions import E2M2EError, PropagationFailure
from e2m2e.integrators import propagate_cr3bp_py

pytestmark = pytest.mark.integrator


def _collapsing_state() -> list[float]:
    """步塌缩初态：在月球附近（x ≈ 1-μ）静止释放，自由落体撞向月球。

    近月点引力 1/r³ 主导，自适应步长控制器不断缩步仍无法满足 1e-12 容差，
    h 跌破 ``MIN_STEP·span`` 循环守卫 → 步长塌缩（确定性传播失败）。
    """
    return [1.0 - Datum.DE421.mu + 1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_propagation_failure_is_e2m2e_error_subclass():
    """``PropagationFailure`` 是 ``E2M2EError`` 子类，支持 ``except E2M2EError`` 统一捕获。"""
    assert issubclass(PropagationFailure, E2M2EError)
    assert isinstance(PropagationFailure("step size collapsed"), E2M2EError)


def test_propagation_failure_not_runtime_error():
    """``PropagationFailure`` 不是 ``RuntimeError`` 子类：步塌缩与通用运行时错误区分。

    这是 ADR 0020 决策 2 类型化的核心——下游用 ``except PropagationFailure``
    精确捕获传播失败，不再与 ``RuntimeError`` 混淆。
    """
    assert not issubclass(PropagationFailure, RuntimeError)


def test_step_collapse_raises_propagation_failure():
    """CR3BP 步塌缩经 FFI 表现为 ``PropagationFailure``。

    复用 Rust 侧 ``step_collapse_early_exit`` 测试的范式（确定发散的初值
    在有限 t_span 内步长塌缩），本测试是其 Python 侧对应：失败识别靠异常
    类型而非错误消息字符串前缀匹配。
    """
    t_eval = np.linspace(0.0, 2.0, 21).tolist()
    with pytest.raises(PropagationFailure):
        propagate_cr3bp_py(
            mu=Datum.DE421.mu,
            t_span=(0.0, 2.0),
            t_eval=t_eval,
            initial_state=_collapsing_state(),
            rtol=1e-12,
            atol=1e-12,
        )


def test_propagate_state_only_returns_empty_states_on_collapse():
    """回归：``_propagate_state_only`` 步塌缩仍返回空 states。

    ADR 0020 迁移第 1 步零行为变更——catch 机制从字符串匹配改为类型匹配
    （``except PropagationFailure``），返回值语义不变（空 states）。下游 NLP
    的 ``len(states)==0`` 嗅探与 ``dv=1e10`` 惩罚逻辑本步不受影响；改成带
    failure 标记的结构化返回是迁移第 4 步。
    """
    system = CR3BP_System(
        mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
    )._with_default_scales()
    dynamics = CR3BP_Dynamics(system)

    result = dynamics._propagate_state_only(
        initial_state=np.array(_collapsing_state(), dtype=float),
        t_span=(0.0, 2.0),
        t_eval=np.linspace(0.0, 2.0, 21),
        max_step=dynamics.max_step,
        with_jacobi=False,
        events=None,
    )

    assert result["time"].size == 0
    assert result["states"].shape == (0, 6)
