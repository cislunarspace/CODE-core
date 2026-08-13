"""Cowell（Störmer-Cowell）八阶二重积分器测试。

覆盖 J2 归一化加速度、启动历史与位置传播。
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from e2m2e.integrators import (
    cowell_step,
    initialize_cowell_history,
)
from tests.numerical.integrators.methods.conftest import EARTH_J2, normalized_leo_j2

pytestmark = pytest.mark.integrator


def _j2_accel_normalised(t: float, x: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """二体 + J2 加速度（仅位置的函数；归一化：mu = re = 1）。

    与 conftest.j2_rhs 的加速度分量一致，保证 Cowell（积 x'' = a(t, x)）
    与 DOP853 对照的是同一份物理。"""
    r = np.asarray(x, dtype=float)
    r_norm = np.linalg.norm(r)
    r2 = r_norm**2
    a_2body = -r / r_norm**3
    k = 1.5 * EARTH_J2 / r_norm**5
    z2_over_r2 = r[2] ** 2 / r2
    a_j2 = -k * np.array(
        [
            r[0] * (1.0 - 5.0 * z2_over_r2),
            r[1] * (1.0 - 5.0 * z2_over_r2),
            r[2] * (3.0 - 5.0 * z2_over_r2),
        ]
    )
    return a_2body + a_j2


def _propagate_cowell(accel, x0, v0, h, target_t, t0=0.0, tol=1e-12):
    """定步长 Cowell 位置传播到 ``target_t``，返回 (t, x)。"""
    t, x, _v, history = initialize_cowell_history(t0, x0, v0, h, accel, tol=tol)
    n_steps = int(round((target_t - t) / h))
    for _ in range(n_steps):
        result = cowell_step(t, h, tol, accel, history)
        x = np.asarray(result.x_new, dtype=float)
        t += h
        history = result.history
    return t, x


def test_cowell_step_history_length_validation():
    """cowell_step 拒绝非 10 向量的 history。"""
    accel = lambda t, x: -x  # noqa: E731

    # Cowell 需要 10 个样本（2 个位置 + 8 个加速度）；传 5 个 → 报错。
    with pytest.raises(ValueError):
        cowell_step(0.0, 0.1, 1e-12, accel, [[1.0]] * 5)


def test_initialize_cowell_history_fills_ten_samples():
    """7 个 RK89 启动步 + 初始加速度 = 10 向量的 history。"""
    accel = lambda t, x: -x  # noqa: E731
    x0 = np.array([1.0])
    v0 = np.array([0.0])
    h = 0.01

    t, x, v, history = initialize_cowell_history(0.0, x0, v0, h, accel)
    assert len(history) == 10
    assert all(len(sample) == 1 for sample in history)
    assert abs(t - 7 * h) < 1e-12
    # 谐振运动 7 步后 x ≈ cos(7h)。
    assert abs(x[0] - np.cos(7 * h)) < 1e-8


def test_initialize_cowell_history_rejects_short_startup():
    """n_startup < 7 无法填满 8 个加速度样本。"""
    accel = lambda t, x: -x  # noqa: E731
    with pytest.raises(ValueError):
        initialize_cowell_history(0.0, np.array([1.0]), np.array([0.0]), 0.01, accel, n_startup=3)


def test_cowell_harmonic_matches_analytic():
    """Cowell 传播谐振子，与解析解一致。"""

    def accel(t, x):  # noqa: ARG001
        return -np.asarray(x, dtype=float)

    x0 = np.array([1.0])
    v0 = np.array([0.0])
    h = 0.01
    t_final = 1.0

    _, x = _propagate_cowell(accel, x0, v0, h, t_final)
    assert abs(x[0] - np.cos(t_final)) < 1e-9


def test_cowell_leo_j2_matches_dop853():
    """Cowell 定步长传播约 1 天，与 scipy DOP853 位置一致（< 1e-9）。

    Cowell 只出位置且定步长，落在目标最近的 h 整数倍处；对照用 DOP853
    稠密输出在 Cowell 实际到达的 t 处取位置。
    """
    rhs, y0, t_span = normalized_leo_j2(days=1.0)
    x0, v0 = y0[:3].copy(), y0[3:].copy()
    accel = _j2_accel_normalised

    h = 0.04  # 归一化时间单位（400 km LEO 约 32 s）
    t_cw, x_cw = _propagate_cowell(accel, x0, v0, h, t_span[1])

    sol = solve_ivp(rhs, t_span, y0, method="DOP853", rtol=1e-12, atol=1e-12, dense_output=True)
    assert sol.success
    x_ref = np.asarray(sol.sol(t_cw))[:3]
    assert np.linalg.norm(x_cw - x_ref) < 1e-9
