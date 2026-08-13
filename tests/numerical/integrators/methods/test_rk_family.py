"""RK 方法族端到端对比测试。

三个方法（PD45/PD78/RK89）共用一套自适应传播封装（本模块的
propagate_rk），在圆轨道二体与归一化 LEO+J2 上与解析解 / scipy DOP853
对照，并验证方法间互洽与“高阶方法步数更少”。
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from e2m2e.integrators import RkMethod, rk_step
from tests.numerical.integrators.methods.conftest import normalized_leo_j2

pytestmark = pytest.mark.integrator


# ---------------------------------------------------------------------------
# 自适应传播封装与开普勒解析解（仅本模块使用）
# ---------------------------------------------------------------------------


def propagate_rk(method, rhs, y0, t_span, tol: float = 1e-12, h0: float = 1.0):
    """经 ``rk_step`` 自适应传播，从 t_span[0] 到 t_span[1]。

    ``tol`` 是*相对*容差：每步接受阈值为 ``tol * max(1, ||y||)``，使控制器
    在不同状态量级下（如归一化二体 vs km 单位 LEO）行为一致。局部误差估计
    超阈值的步被拒，按 ``rk_step`` 建议的更小步长重试。

    返回 ``(t_final, y_final, n_steps)``。
    """
    t0, tf = t_span
    t = float(t0)
    y = np.asarray(y0, dtype=float).copy()
    h = float(h0)
    n_steps = 0
    while t < tf:
        abs_tol = tol * max(1.0, float(np.linalg.norm(y)))
        h_step = min(h, tf - t)
        result = rk_step(method, t, y, h_step, abs_tol, rhs)
        if result.error <= abs_tol:
            # 接受；步长增长限 2 倍，避免接受时误差过冲。
            y = np.asarray(result.y_new, dtype=float)
            t += h_step
            h = min(result.h_next, h_step * 2.0)
        else:
            # 拒绝：y/t 不动，按建议的更小步长重试。
            h = result.h_next
        n_steps += 1
    return t, y, n_steps


def solve_kepler(M: float, e: float, tol: float = 1e-14, max_iter: int = 100) -> float:
    """解开普勒方程 M = E - e*sin(E)，求偏近点角 E。"""
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E = E + dE
        if abs(dE) < tol:
            break
    return E


def kepler_analytic_state(r0: np.ndarray, v0: np.ndarray, t: float, mu: float = 1.0) -> np.ndarray:
    """用开普勒方程解析传播二体初值，返回 t 时刻状态向量 [x, y, z, vx, vy, vz]。"""
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    r0_norm = np.linalg.norm(r0)
    v0_norm = np.linalg.norm(v0)

    # 比轨道能量与角动量
    energy = 0.5 * v0_norm**2 - mu / r0_norm
    h_vec = np.cross(r0, v0)
    h = np.linalg.norm(h_vec)

    # 半长轴
    a = -mu / (2.0 * energy)

    # 偏心率矢量
    e_vec = ((v0_norm**2 - mu / r0_norm) * r0 - np.dot(r0, v0) * v0) / mu
    e = np.linalg.norm(e_vec)

    # 平均运动
    n = np.sqrt(mu / a**3)

    # 近圆轨道直接处理，避免除以小偏心率
    if e < 1e-12:
        # 轨道面内的圆周运动
        z_hat = h_vec / h
        x_hat = r0 / r0_norm
        y_hat = np.cross(z_hat, x_hat)

        theta0 = np.arctan2(r0[1], r0[0])
        theta = theta0 + n * t

        r_t = a * (np.cos(theta) * x_hat + np.sin(theta) * y_hat)
        v_t = n * a * (-np.sin(theta) * x_hat + np.cos(theta) * y_hat)
        return np.concatenate([r_t, v_t])

    # 由初始条件求偏近点角
    E0 = np.arccos(np.clip((1.0 - r0_norm / a) / e, -1.0, 1.0))
    if np.dot(r0, v0) < 0.0:
        E0 = 2.0 * np.pi - E0

    # 历元平近点角
    M0 = E0 - e * np.sin(E0)

    # t 时刻平近点角
    M = M0 + n * t
    M = np.mod(M, 2.0 * np.pi)

    # 解开普勒方程求 E
    E = solve_kepler(M, e)

    # 真近点角
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0),
    )

    # 轨道面基矢量
    z_hat = h_vec / h
    x_hat = e_vec / e if e > 1e-12 else r0 / r0_norm
    y_hat = np.cross(z_hat, x_hat)

    # 轨道面内位置与速度
    p = a * (1.0 - e**2)
    r = p / (1.0 + e * np.cos(nu))

    r_orbit = r * np.array([np.cos(nu), np.sin(nu)])
    v_orbit = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu)])

    r_t = r_orbit[0] * x_hat + r_orbit[1] * y_hat
    v_t = v_orbit[0] * x_hat + v_orbit[1] * y_hat

    return np.concatenate([r_t, v_t])


# ---------------------------------------------------------------------------
# 方法参数集
# ---------------------------------------------------------------------------

RK_METHODS = [
    pytest.param(RkMethod.PD45, id="PD45"),
    pytest.param(RkMethod.PD78, id="PD78"),
    pytest.param(RkMethod.RK89, id="RK89"),
]

# (method, tol, 解析对照阈值) —— 高阶方法以更高精度结束。
CIRCULAR_CASES = [
    pytest.param(RkMethod.PD45, 1e-12, 1e-9, id="PD45"),
    pytest.param(RkMethod.PD78, 1e-12, 1e-10, id="PD78"),
    pytest.param(RkMethod.RK89, 1e-13, 1e-11, id="RK89"),
]


def _two_body(t, y):  # noqa: ARG001
    """归一化二体加速度（mu=1）。"""
    r = y[:3]
    v = y[3:]
    r_norm = np.linalg.norm(r)
    a = -r / r_norm**3
    return np.concatenate([v, a])


# ---------------------------------------------------------------------------
# 圆轨道二体
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "tol", "atol"), CIRCULAR_CASES)
def test_two_body_circular_matches_analytic(method, tol, atol):
    """整圈圆轨道传播与开普勒解析解对照。"""
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
    period = 2.0 * np.pi
    _, y_final, _ = propagate_rk(method, _two_body, y0, (0.0, period), tol=tol, h0=0.01)

    y_exact = kepler_analytic_state(y0[:3], y0[3:], period)
    assert np.linalg.norm(y_final - y_exact) < atol


def test_pd45_circular_matches_scipy():
    """PD45 整圈圆轨道与 scipy RK45 对照（独立基准）。

    手写自适应循环（不借助 propagate_rk），同时验证
    手工编排 rk_step 的用法契约。
    """
    from scipy.integrate import solve_ivp

    y0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
    period = 2.0 * np.pi
    tol = 1e-12

    # Rust PD45：手工自适应步长接受循环
    t = 0.0
    y = y0.copy()
    h = 0.01
    while t < period:
        h = min(h, period - t)
        result = rk_step(RkMethod.PD45, t, y, h, tol, _two_body)
        y = np.asarray(result.y_new)
        t += h
        h = result.h_next

    # scipy RK45 基准
    sol = solve_ivp(_two_body, (0.0, period), y0, method="RK45", rtol=tol, atol=tol)
    y_scipy = sol.y[:, -1]

    assert np.linalg.norm(y - y_scipy) < 1e-9


# ---------------------------------------------------------------------------
# 归一化 LEO+J2 一天传播
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", RK_METHODS)
def test_leo_j2_matches_dop853(method):
    """归一化 LEO+J2 一天传播与 scipy DOP853 对照（< 1e-9）。

    归一化单位（DU = 地球半径，TU s.t. mu = 1）保持 ||y|| ~ O(1)，
    使相对容差直接控制误差。
    """
    rhs, y0, t_span = normalized_leo_j2(days=1.0)
    sol = solve_ivp(rhs, t_span, y0, method="DOP853", rtol=1e-13, atol=1e-13, dense_output=True)
    assert sol.success

    t, y, _ = propagate_rk(method, rhs, y0, t_span, tol=1e-13, h0=0.01)
    y_ref = np.asarray(sol.sol(t))
    err = np.linalg.norm(y - y_ref)
    assert err < 1e-9, f"{method} 与 DOP853 误差 {err} 过大"


# ---------------------------------------------------------------------------
# 方法间关系
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("m1", "m2"),
    [
        pytest.param(RkMethod.PD45, RkMethod.PD78, id="PD45-PD78"),
        pytest.param(RkMethod.PD78, RkMethod.RK89, id="PD78-RK89"),
        pytest.param(RkMethod.PD45, RkMethod.RK89, id="PD45-RK89"),
    ],
)
def test_rk_family_mutual_consistency(m1, m2):
    """方法两两互洽（< 1e-9）。"""
    rhs, y0, t_span = normalized_leo_j2(days=1.0)
    _, y1, _ = propagate_rk(m1, rhs, y0, t_span, tol=1e-13, h0=0.01)
    _, y2, _ = propagate_rk(m2, rhs, y0, t_span, tol=1e-13, h0=0.01)

    assert np.linalg.norm(y1 - y2) < 1e-9


def test_higher_order_uses_fewer_steps():
    """PD78（8 阶）与 RK89（9 阶）步数远少于 PD45（5 阶）。

    RK89（Verner 9(8)）携带更大的误差常数，在此容差带内步数未必
    低于 PD78，故只断言高阶对与 5 阶之间无歧义的差距。
    """
    rhs, y0, t_span = normalized_leo_j2(days=1.0)
    _, _, n_pd45 = propagate_rk(RkMethod.PD45, rhs, y0, t_span, tol=1e-13, h0=0.01)
    _, _, n_pd78 = propagate_rk(RkMethod.PD78, rhs, y0, t_span, tol=1e-13, h0=0.01)
    _, _, n_rk89 = propagate_rk(RkMethod.RK89, rhs, y0, t_span, tol=1e-13, h0=0.01)

    assert n_pd78 < n_pd45 / 5
    assert n_rk89 < n_pd45 / 5
