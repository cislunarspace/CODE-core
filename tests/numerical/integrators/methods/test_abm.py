"""ABM（Adams-Bashforth-Moulton）多步积分器测试。

覆盖历史长度校验、启动填充、谐振子精度与四阶收敛。
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from e2m2e.integrators import (
    MultistepMethod,
    initialize_abm_history,
    multistep_step,
)
from tests.numerical.integrators.methods.conftest import normalized_leo_j2

pytestmark = pytest.mark.integrator


def _propagate_abm(f, y0, h, target_t, t0=0.0):
    """定步长 ABM 传播到 ``target_t``，返回 (t, y)。"""
    t, y, history = initialize_abm_history(t0, y0, h, f, n_stages=3)
    n_steps = int(round((target_t - t) / h))
    for _ in range(n_steps):
        result = multistep_step(MultistepMethod.ABM, t, y, h, 1e-12, f, history)
        y = np.asarray(result.y_new, dtype=float)
        t += h
        history = result.history
    return t, y


def test_multistep_step_history_length_validation():
    """multistep_step 拒绝长度不符的 history。"""
    y0 = np.array([1.0, 0.0])

    def f(t, y):  # noqa: ARG001
        return np.array([y[1], -y[0]])

    # ABM 需要 4 个导数样本；传 3 个 → 报错。
    with pytest.raises(ValueError):
        multistep_step(MultistepMethod.ABM, 0.0, y0, 0.1, 1e-12, f, [[0, -1]] * 3)


def test_initialize_abm_history_fills_four_samples():
    """3 个 RK89 启动步 + 初始导数 = 4 个样本的 history。"""
    y0 = np.array([1.0, 0.0])

    def f(t, y):  # noqa: ARG001
        return np.array([y[1], -y[0]])

    h = 0.01
    t, y, history = initialize_abm_history(0.0, y0, h, f, n_stages=3)
    assert len(history) == 4
    assert all(len(sample) == 2 for sample in history)
    assert abs(t - 3 * h) < 1e-12
    # 谐振运动 3 步后 y ≈ [cos(3h), -sin(3h)]。
    assert abs(y[0] - np.cos(3 * h)) < 1e-6


def test_abm_harmonic_matches_analytic():
    """ABM 传播谐振子，与解析解一致。"""

    def f(t, y):  # noqa: ARG001
        return np.array([y[1], -y[0]])

    y0 = np.array([1.0, 0.0])
    h = 0.005
    t_final = 1.0
    t, y = _propagate_abm(f, y0, h, t_final)

    exact = np.array([np.cos(t_final), -np.sin(t_final)])
    assert np.linalg.norm(y - exact) < 1e-5


def test_abm_convergence_is_fourth_order():
    """步长减半误差缩小 ~2^4 = 16 倍（4 阶）。"""

    def f(t, y):  # noqa: ARG001
        return np.array([y[1], -y[0]])

    y0 = np.array([1.0, 0.0])
    target_t = 1.0
    errors = []
    for h in (0.02, 0.01):
        _, y = _propagate_abm(f, y0, h, target_t)
        exact = np.array([np.cos(target_t), -np.sin(target_t)])
        errors.append(np.linalg.norm(y - exact))

    ratio = errors[0] / errors[1]
    assert 10.0 < ratio < 30.0, f"收敛比 {ratio} 不在 ~16（4 阶）附近"


def test_abm_leo_j2_matches_dop853():
    """ABM 小定步长传播约 1 天，与 scipy DOP853 一致（< 1e-6）。

    ABM 是定步长，落在目标最近的 h 整数倍处；对照用 DOP853 稠密输出
    在 ABM 实际到达的 t 处取值。
    """
    rhs, y0, t_span = normalized_leo_j2(days=1.0)
    h = 0.002
    t_abm, y_abm = _propagate_abm(rhs, y0, h, t_span[1])

    sol = solve_ivp(rhs, t_span, y0, method="DOP853", rtol=1e-12, atol=1e-12, dense_output=True)
    assert sol.success
    y_ref = np.asarray(sol.sol(t_abm))
    assert np.linalg.norm(y_abm - y_ref) < 1e-6


def test_abm_step_size_change_requires_restart():
    """不换 history 直接改步长会发散。

    陈旧（间距错误）的 history 会让预测器吃入垃圾导数样本，结果偏离
    重新初始化后的参考很远。本测试记录定步长契约，不断言具体数值。
    """
    rhs, y0, t_span = normalized_leo_j2(days=0.2)
    h = 0.01

    # 正确用法：传播中途改步长必须重新初始化 history。
    t, y, history = initialize_abm_history(0.0, y0, h, rhs, n_stages=3)
    for _ in range(5):
        r = multistep_step(MultistepMethod.ABM, t, y, h, 1e-12, rhs, history)
        y, t, history = np.asarray(r.y_new), t + h, r.history
    # 现在步长减半：必须按新间距重建 history。
    h2 = h / 2
    _, y_restarted, hist2 = initialize_abm_history(t, y, h2, rhs, n_stages=3)

    # 错误用法：新步长沿用旧的（h 间距）history。
    r_stale = multistep_step(MultistepMethod.ABM, t, y, h2, 1e-12, rhs, history)

    # 重新初始化的一步与陈旧 history 的一步应显著不同。
    r_good = multistep_step(MultistepMethod.ABM, t, y_restarted, h2, 1e-12, rhs, hist2)
    assert np.linalg.norm(np.asarray(r_stale.y_new) - np.asarray(r_good.y_new)) > 1e-8
