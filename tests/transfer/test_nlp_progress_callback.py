"""
需求: DROTRONLPOptimizer 进度回调

在 NLP 优化（SLSQP 迭代）过程中，调用方需要实时获知迭代进度（当前迭代序号、
目标函数值、优化变量 α/T/t_ins），以便在批量优化场景下显示进度条或写入日志。

验收标准:
  1. DROTRONLPOptimizer 新增 set_progress_callback(callback) 方法
  2. callback 签名: callback(iteration: int, objective: float,
                             alpha: float, transfer_time: float, t_ins: float) -> None
  3. 每次 SLSQP 迭代结束后调用 callback
  4. 不设置 callback 时行为与当前完全一致（不报错、无副作用）
  5. callback 为 None 时等价于未设置

参考论文: Cui et al. (2025), Section III.B
"""

import numpy as np
import pytest

from e2m2e.core import Orbit, CR3BP_System, CR3BP_Dynamics
from e2m2e.transfer import DROTRONLPOptimizer, NLPOptimizationVariables


def _simple_orbit(n: int = 80) -> Orbit:
    t = np.linspace(0, 6.28, n)
    x = 0.9 + 0.08 * np.cos(t)
    y = 0.08 * np.sin(t)
    z = np.zeros_like(t)
    vx = -0.08 * np.sin(t)
    vy = 0.08 * np.cos(t)
    vz = np.zeros_like(t)
    states = np.column_stack([x, y, z, vx, vy, vz])
    orbit = Orbit(states, t)
    orbit.period = float(t[-1])
    return orbit


@pytest.fixture
def system():
    return CR3BP_System(mu=1.21506683e-2, primary="earth", secondary="moon")


@pytest.fixture
def dynamics(system):
    d = CR3BP_Dynamics(system)
    d.integrator = "RK45"
    d.rtol = d.atol = 1e-3
    d.max_step = 2.0
    return d


@pytest.fixture
def optimizer(system, dynamics):
    dro = _simple_orbit(60)
    ro = _simple_orbit(50)
    dep = dro.states[0]
    return DROTRONLPOptimizer(
        system=system,
        dynamics=dynamics,
        departure_orbit=dro,
        arrival_orbit=ro,
        departure_state=dep,
    )


class TestSetProgressCallback:
    def test_method_exists(self, optimizer):
        assert hasattr(optimizer, "set_progress_callback")
        assert callable(optimizer.set_progress_callback)

    def test_accepts_callable(self, optimizer):
        records = []

        def cb(it, obj, alpha, T, tins):
            records.append((it, obj, alpha, T, tins))

        optimizer.set_progress_callback(cb)
        assert optimizer._progress_callback is cb

    def test_accepts_none(self, optimizer):
        optimizer.set_progress_callback(None)
        assert optimizer._progress_callback is None

    def test_no_callback_no_error(self, optimizer):
        optimizer.set_progress_callback(None)
        result = optimizer.optimize(
            initial_guess=NLPOptimizationVariables(alpha=1.0, transfer_time=5.0, t_ins=3.0),
            alpha_range=(0.5, 2.5),
            transfer_time_range=(1.0, 20.0),
            t_ins_range=(0.0, 6.0),
            verbose=False,
        )
        assert result is not None


class TestCallbackInvoked:
    def test_callback_receives_iterations(self, optimizer):
        records = []

        def cb(it, obj, alpha, T, tins):
            records.append((it, obj, alpha, T, tins))

        optimizer.set_progress_callback(cb)
        result = optimizer.optimize(
            initial_guess=NLPOptimizationVariables(alpha=1.0, transfer_time=5.0, t_ins=3.0),
            alpha_range=(0.5, 2.5),
            transfer_time_range=(1.0, 20.0),
            t_ins_range=(0.0, 6.0),
            verbose=False,
        )

        assert len(records) >= 1, "callback 应至少被调用一次"

    def test_callback_arguments_type_and_shape(self, optimizer):
        records = []

        def cb(it, obj, alpha, T, tins):
            records.append((it, obj, alpha, T, tins))

        optimizer.set_progress_callback(cb)
        optimizer.optimize(
            initial_guess=NLPOptimizationVariables(alpha=1.0, transfer_time=5.0, t_ins=3.0),
            alpha_range=(0.5, 2.5),
            transfer_time_range=(1.0, 20.0),
            t_ins_range=(0.0, 6.0),
            verbose=False,
        )

        for it, obj, alpha, T, tins in records:
            assert isinstance(it, int) or isinstance(it, np.integer), (
                f"iteration 应为 int, 实际 {type(it)}"
            )
            assert isinstance(obj, float) or isinstance(obj, np.floating), (
                f"objective 应为 float, 实际 {type(obj)}"
            )
            assert isinstance(alpha, float) or isinstance(alpha, np.floating)
            assert isinstance(T, float) or isinstance(T, np.floating)
            assert isinstance(tins, float) or isinstance(tins, np.floating)

    def test_iteration_numbers_monotonically_increase(self, optimizer):
        records = []

        def cb(it, obj, alpha, T, tins):
            records.append(it)

        optimizer.set_progress_callback(cb)
        optimizer.optimize(
            initial_guess=NLPOptimizationVariables(alpha=1.0, transfer_time=5.0, t_ins=3.0),
            alpha_range=(0.5, 2.5),
            transfer_time_range=(1.0, 20.0),
            t_ins_range=(0.0, 6.0),
            verbose=False,
        )

        for i in range(1, len(records)):
            assert records[i] >= records[i - 1], (
                f"迭代序号应单调递增: {records[i - 1]} -> {records[i]}"
            )

    def test_objective_is_finite(self, optimizer):
        records = []

        def cb(it, obj, alpha, T, tins):
            records.append(obj)

        optimizer.set_progress_callback(cb)
        optimizer.optimize(
            initial_guess=NLPOptimizationVariables(alpha=1.0, transfer_time=5.0, t_ins=3.0),
            alpha_range=(0.5, 2.5),
            transfer_time_range=(1.0, 20.0),
            t_ins_range=(0.0, 6.0),
            verbose=False,
        )

        for obj in records:
            assert np.isfinite(obj), f"objective 应为有限值, 实际 {obj}"

    def test_alpha_within_bounds(self, optimizer):
        records = []

        def cb(it, obj, alpha, T, tins):
            records.append(alpha)

        optimizer.set_progress_callback(cb)
        optimizer.optimize(
            initial_guess=NLPOptimizationVariables(alpha=1.0, transfer_time=5.0, t_ins=3.0),
            alpha_range=(0.5, 2.5),
            transfer_time_range=(1.0, 20.0),
            t_ins_range=(0.0, 6.0),
            verbose=False,
        )

        for alpha in records:
            assert 0.5 <= alpha <= 2.5, f"alpha 应在 [0.5, 2.5] 范围内, 实际 {alpha}"


class TestCallbackDoesNotAffectResult:
    def test_optimize_result_unchanged_with_callback(self, optimizer):
        optimizer.set_progress_callback(None)
        result_no_cb = optimizer.optimize(
            initial_guess=NLPOptimizationVariables(alpha=1.0, transfer_time=5.0, t_ins=3.0),
            alpha_range=(0.5, 2.5),
            transfer_time_range=(1.0, 20.0),
            t_ins_range=(0.0, 6.0),
            verbose=False,
        )

        records = []

        def cb(it, obj, alpha, T, tins):
            records.append(1)

        optimizer.set_progress_callback(cb)
        result_with_cb = optimizer.optimize(
            initial_guess=NLPOptimizationVariables(alpha=1.0, transfer_time=5.0, t_ins=3.0),
            alpha_range=(0.5, 2.5),
            transfer_time_range=(1.0, 20.0),
            t_ins_range=(0.0, 6.0),
            verbose=False,
        )

        assert result_with_cb.success == result_no_cb.success
        assert np.isclose(result_with_cb.alpha, result_no_cb.alpha, atol=1e-4)
        assert np.isclose(
            result_with_cb.objective_value,
            result_no_cb.objective_value,
            atol=1e-4,
        )
