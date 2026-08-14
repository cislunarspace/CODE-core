"""ADR 0020 决策 2（#353）：传播失败显式标记 + NLP 去目标惩罚。

覆盖三组行为：

1. 直接 ``propagate()`` 在步塌缩时抛 ``PropagationFailure``；
   ``DROTRONLPOptimizer.forward_integrate()`` 在搜索语境将其翻译为
   ``status``/``cause``，不把空 states 暴露为传播接口契约；
2. ``DROTRONLPOptimizer._evaluate_all`` 对不可行候选读标记而非 ``len==0``
   嗅探，目标不被 1e10/2e10 惩罚污染，不可行由约束冲突 + ``INFEASIBLE`` 表达；
3. SLSQP/COPT 共享同一组回调（``objective_function`` /
   ``constraint_position`` / ``constraint_velocity_parallel``），不可行语义
   在两后端一致（与 ``test_rust_backend_equivalence`` 同一对照精神）。

测试构造：用极小 ``max_step`` 强制 CR3BP 步长塌缩，制造确定性的传播失败；
优化器在候选评估接缝捕获 ``PropagationFailure`` 并转 ``DIVERGED`` 标记。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.transfer import DROTRONLPOptimizer
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


def _make_system() -> CR3BP_System:
    return CR3BP_System(mu=0.012150585, primary="earth", secondary="moon")


def _make_dynamics(max_step: float = 1e-300) -> CR3BP_Dynamics:
    dyn = CR3BP_Dynamics(system=_make_system())
    # 极小 max_step 使积分器步长立即塌缩（Rust 路径抛 PropagationFailure）。
    dyn.max_step = max_step
    return dyn


def _make_orbit(system: CR3BP_System) -> Orbit:
    """构造一条合理初态的周期轨道（非奇异，速度非零）。"""
    state0 = np.array([1.12, 0.0, 0.0, 0.0, 0.2, 0.0])
    orbit = Orbit(
        states=np.tile(state0, (10, 1)),
        times=np.linspace(0.0, 10.0, 10),
        system=system,
    )
    orbit.period = 10.0
    return orbit


def _make_optimizer(dynamics: CR3BP_Dynamics) -> DROTRONLPOptimizer:
    system = _make_system()
    dep = _make_orbit(system)
    arr = _make_orbit(system)
    return DROTRONLPOptimizer.from_orbits(system, dynamics, dep, arr)


class TestOptimizerPropagationFailureTranslation:
    """优化器在搜索语境翻译传播失败（ADR 0020 决策 1、2）。"""

    def test_step_collapse_becomes_a_diverged_candidate(self):
        optimizer = _make_optimizer(_make_dynamics())
        times, states = optimizer.forward_integrate(
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (0.0, 5.0),
            t_eval=np.array([5.0]),
        )
        assert times.size == 0
        assert states.shape == (0, 6)
        assert optimizer._last_prop_status is ConvergenceState.DIVERGED
        assert optimizer._last_prop_cause is FailureCause.DIVERGENCE_DETECTED

    def test_successful_candidate_keeps_converged_status(self):
        optimizer = _make_optimizer(CR3BP_Dynamics(system=_make_system()))
        _, states = optimizer.forward_integrate(
            np.array([0.8, 0.0, 0.0, 0.0, 0.05, 0.0]),
            (0.0, 0.5),
            t_eval=np.array([0.5]),
        )
        assert states.shape == (1, 6)
        assert optimizer._last_prop_status is ConvergenceState.CONVERGED
        assert optimizer._last_prop_cause is FailureCause.NONE


class TestEvaluateAllInfeasibleCandidate:
    """``_evaluate_all`` 读标记、去目标惩罚、留约束冲突（#353）。"""

    def test_infeasible_candidate_markers(self):
        """传播失败候选：INFEASIBLE + inf 目标 + 1e6 约束冲突，无 1e10 惩罚。"""
        opt = _make_optimizer(_make_dynamics())
        y = np.array([1.0, 10.0, 5.0])  # [alpha, transfer_time, t_ins]
        cache = opt._evaluate_all(y)

        assert cache["empty"] is True
        assert cache["status"] is ConvergenceState.INFEASIBLE
        assert cache["cause"] is FailureCause.DIVERGENCE_DETECTED
        # 目标不被惩罚值污染（ADR 0020 决策 2）
        assert cache["objective"] == float("inf")
        assert not np.isfinite(cache["dv1"])
        assert not np.isfinite(cache["dv2"])
        # 约束冲突保留有限大值作为不可行信号
        assert cache["pos_violation"] == 1e6
        assert cache["vel_constraint"] == 1e6
        # 魔法惩罚值不再出现
        assert cache["objective"] != 2e10
        assert cache["dv1"] != 1e10

    def test_objective_function_infeasible_returns_inf(self):
        """objective_function 对传播失败候选返回 inf，而非 1e10。"""
        opt = _make_optimizer(_make_dynamics())
        y = np.array([1.0, 10.0, 5.0])
        assert opt.objective_function(y) == float("inf")

    def test_feasible_candidate_objective_not_polluted(self):
        """可行候选目标值正常（回归：惩罚改动不影响可行路径）。"""
        opt = _make_optimizer(_make_dynamics(max_step=1e-3))
        y = np.array([1.0, 1.0, 0.5])
        cache = opt._evaluate_all(y)
        assert cache["empty"] is False
        assert cache["status"] is ConvergenceState.CONVERGED
        assert np.isfinite(cache["objective"])


class TestSlsqpCoptSharedCallbacks:
    """SLSQP/COPT 共享回调的不可行语义一致性。

    ``nlp_scipy`` 与 ``nlp_copt`` 都消费 ``objective_function`` /
    ``constraint_position`` / ``constraint_velocity_parallel``（见
    ``transfer_optimization.py`` 与 ``nlp_scipy.py``/``nlp_copt.py``），
    故不可行候选在两后端表现一致——同一实现、同一返回值。
    """

    def test_shared_callbacks_agree_on_infeasible(self):
        opt = _make_optimizer(_make_dynamics())
        y = np.array([1.0, 10.0, 5.0])
        # 目标（objective_function）与两个约束（constraint_position /
        # constraint_velocity_parallel）是 nlp_scipy / nlp_copt 共用的回调：
        # 不可行候选在三个回调上表达一致（inf / 1e6 / 1e6），两后端无分叉。
        assert opt.objective_function(y) == float("inf")
        assert opt.constraint_position(y) == 1e6
        assert opt.constraint_velocity_parallel(y) == 1e6
