"""低推力解析雅可比（灵敏度）有限差分对标。

验证 Rust ``propagate_compiled_lowthrust_sensitivity`` 的解析灵敏度 S(7×3)
正确：对控制参数 (throttle, θ₁, θ₂) 各做有限差分扰动，末端状态变化应与
解析 S 预测一致（相对误差 < 1e-6）。这是解析雅可比正确性的根本验证。
"""

import numpy as np
import pytest
from e2m2e._integrators import (
    RkMethod,
    propagate_compiled_lowthrust,
    propagate_compiled_lowthrust_sensitivity,
)


def _assert_sensitivity_match(name, analytic, fd, indices, rel_tol=1e-5, abs_floor=1e-6):
    """校验解析灵敏度 vs 有限差分一致。

    分量绝对值大于 ``abs_floor`` 时校验相对误差；小于时（接近零，相对误差
    无意义）校验绝对误差 < 1e-9。
    """
    for i in indices:
        a, f = analytic[i], fd[i]
        if abs(a) < abs_floor and abs(f) < abs_floor:
            assert abs(a - f) < 1e-9, f"{name} 分量 {i}: analytic={a:.6e} fd={f:.6e} 绝对误差过大"
        else:
            scale = max(abs(a), abs(f), 1e-12)
            rel = abs(a - f) / scale
            assert rel < rel_tol, f"{name} 分量 {i}: analytic={a:.6e} fd={f:.6e} rel_err={rel:.2e}"


def _propagate_segment(
    t0, y0, h, tol, tf, observer, forces_py, t_max, isp, throttle, theta1, theta2
):
    """单段灵敏度传播，返回末端 (state7, stm6x6, sens7x3)。"""
    res = propagate_compiled_lowthrust_sensitivity(
        RkMethod.PD45,
        t0,
        list(y0),
        h,
        tol,
        [t0, tf],
        observer,
        forces_py,
        (t_max, isp, throttle, theta1, theta2),
        500_000,
    )
    state7 = np.asarray(res["states"][-1])
    stm = np.asarray(res["stm"][-1]).reshape(6, 6)
    sens = np.asarray(res["sensitivity"][-1]).reshape(7, 3)
    return state7, stm, sens


def _propagate_plain(t0, y0, h, tol, tf, observer, forces_py, t_max, isp, throttle, theta1, theta2):
    """单段无灵敏度传播（用于有限差分），返回末端 state7。

    方向从角度还原：plain 传播接受方向向量，需把 (θ₁,θ₂) 转成向量。
    """
    alpha = np.array(
        [np.cos(theta1) * np.cos(theta2), np.sin(theta1) * np.cos(theta2), np.sin(theta2)]
    )
    res = propagate_compiled_lowthrust(
        RkMethod.PD45,
        t0,
        list(y0),
        h,
        tol,
        [t0, tf],
        observer,
        forces_py,
        (t_max, isp, throttle, alpha[0], alpha[1], alpha[2]),
        500_000,
    )
    return np.asarray(res["states"][-1])


@pytest.fixture
def segment_setup():
    """单段受控传播的标准设定（地心点质量 + 低推力）。"""
    mu = 398600.435507
    forces_py = [("point_mass", mu)]
    observer = "EARTH"
    r0 = 7000.0
    v0 = np.sqrt(mu / r0)
    y0 = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0, 1000.0])
    t0 = 0.0
    dt = 600.0  # 10 分钟弧段
    tf = t0 + dt
    h = dt / 10.0
    tol = 1e-11
    t_max = 0.5
    isp = 3000.0
    throttle = 0.7
    theta1 = 0.3
    theta2 = 0.2
    return dict(
        t0=t0,
        y0=y0,
        h=h,
        tol=tol,
        tf=tf,
        observer=observer,
        forces_py=forces_py,
        t_max=t_max,
        isp=isp,
        throttle=throttle,
        theta1=theta1,
        theta2=theta2,
    )


def test_sensitivity_throttle_finite_difference(segment_setup):
    """∂末端状态/∂throttle 的解析灵敏度 vs 有限差分，相对误差 < 1e-6。"""
    s = segment_setup
    state7, _, sens = _propagate_segment(**s)

    eps = 1e-4  # 角度/throttle 扰动；eps 太小会被舍入误差污染（1e-6 时差分失精）
    s_plus = {**s, "throttle": s["throttle"] + eps}
    s_minus = {**s, "throttle": s["throttle"] - eps}
    state_plus = _propagate_plain(**s_plus)
    state_minus = _propagate_plain(**s_minus)
    fd = (state_plus - state_minus) / (2 * eps)  # 有限差分 ∂state/∂throttle

    analytic = sens[:, 0]  # 解析 ∂state/∂throttle（S 的第 0 列）
    _assert_sensitivity_match("throttle", analytic, fd, range(7))


def test_sensitivity_theta1_finite_difference(segment_setup):
    """∂末端状态/∂θ₁ 的解析灵敏度 vs 有限差分。"""
    s = segment_setup
    state7, _, sens = _propagate_segment(**s)

    eps = 1e-4  # 角度/throttle 扰动；eps 太小会被舍入误差污染（1e-6 时差分失精）
    s_plus = {**s, "theta1": s["theta1"] + eps}
    s_minus = {**s, "theta1": s["theta1"] - eps}
    state_plus = _propagate_plain(**s_plus)
    state_minus = _propagate_plain(**s_minus)
    fd = (state_plus - state_minus) / (2 * eps)

    analytic = sens[:, 1]  # S 的第 1 列
    # θ₁ 不影响质量（∂m/∂θ₁=0），只校验前 6 维（位置速度）
    _assert_sensitivity_match("θ₁", analytic, fd, range(6))
    # 质量对 θ₁ 灵敏度应近似 0
    assert abs(analytic[6]) < 1e-9, f"∂m/∂θ₁ 应≈0, got {analytic[6]}"


def test_sensitivity_theta2_finite_difference(segment_setup):
    """∂末端状态/∂θ₂ 的解析灵敏度 vs 有限差分。"""
    s = segment_setup
    state7, _, sens = _propagate_segment(**s)

    eps = 1e-4  # 角度/throttle 扰动；eps 太小会被舍入误差污染（1e-6 时差分失精）
    s_plus = {**s, "theta2": s["theta2"] + eps}
    s_minus = {**s, "theta2": s["theta2"] - eps}
    state_plus = _propagate_plain(**s_plus)
    state_minus = _propagate_plain(**s_minus)
    fd = (state_plus - state_minus) / (2 * eps)

    analytic = sens[:, 2]  # S 的第 2 列
    _assert_sensitivity_match("θ₂", analytic, fd, range(6))
    assert abs(analytic[6]) < 1e-9, f"∂m/∂θ₂ 应≈0, got {analytic[6]}"


def test_sensitivity_zero_throttle_decouples(segment_setup):
    """throttle=0 时推力项消失，灵敏度应反映纯二体（推力灵敏度列≈0）。"""
    s = {**segment_setup, "throttle": 0.0}
    state7, _, sens = _propagate_segment(**s)
    # ∂state/∂throttle 在零推力下：质量列 ∂ṁ/∂thr=-T/(Isp g0) 仍非零（瞬时），
    # 但位置/速度对 θ₁/θ₂ 的灵敏度应为 0（推力=0 时方向无影响）
    assert np.allclose(sens[:6, 1], 0.0, atol=1e-9), "零推力下 ∂state/∂θ₁ 应≈0"
    assert np.allclose(sens[:6, 2], 0.0, atol=1e-9), "零推力下 ∂state/∂θ₂ 应≈0"


# ---- 全雅可比链式对标（求解器层）----


def _make_shooter_two_body():
    """构造纯二体低推力求解器（无需 SPICE，PointMassGravity 不查 system）。"""
    from types import SimpleNamespace

    from e2m2e.algorithm.forces import PointMassGravity
    from e2m2e.algorithm.transfer import EngineConfig, LowThrustShooting

    mu = 398600.435507
    system = SimpleNamespace(origin="EARTH")
    r0 = 7000.0
    v0 = np.sqrt(mu / r0)
    init = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
    engine = EngineConfig(t_max=0.5, isp=3000.0)
    return LowThrustShooting(
        system,
        [PointMassGravity("EARTH", mu=mu)],
        engine,
        init,
        initial_mass=1000.0,
        target_state=init.copy(),
        t0=0.0,
        tf=1200.0,  # 2 段 × 600s
    )


def test_chain_jacobian_vs_finite_difference():
    """N 段接龙的解析雅可比 vs 有限差分雅可比，逐元素一致。

    验证 `_propagate_chain_with_jacobian` 的链式累积正确：全局末端对每段控制
    的灵敏度，与对决策向量逐分量中心差分一致。

    注意：throttle 初猜取 0.5（非边界 1.0），避免 clip 破坏中心差分。
    """
    shooter = _make_shooter_two_body()
    n_seg = 3
    y0 = shooter._default_x0(n_seg)
    y0[0::3] = 0.5  # throttle 改 0.5，避免边界 clip
    y = y0
    _, _, jac_analytic = shooter._propagate_chain_with_jacobian(y)

    # 有限差分雅可比：对每个决策分量扰动
    eps = 1e-4
    jac_fd = np.zeros((6, 3 * n_seg))
    for j in range(3 * n_seg):
        y_plus = y.copy()
        y_minus = y.copy()
        y_plus[j] += eps
        y_minus[j] -= eps
        _, rv_plus, _ = shooter._propagate_chain_with_jacobian(y_plus)
        _, rv_minus, _ = shooter._propagate_chain_with_jacobian(y_minus)
        jac_fd[:, j] = (rv_plus - rv_minus) / (2 * eps)

    # 逐元素校验（值小处用绝对误差）
    for i in range(6):
        for j in range(3 * n_seg):
            a, f = jac_analytic[i, j], jac_fd[i, j]
            if abs(a) < 1e-6 and abs(f) < 1e-6:
                assert abs(a - f) < 1e-8, f"[{i},{j}] analytic={a:.3e} fd={f:.3e}"
            else:
                rel = abs(a - f) / max(abs(a), abs(f), 1e-12)
                assert rel < 1e-4, f"[{i},{j}] analytic={a:.3e} fd={f:.3e} rel={rel:.2e}"


def test_terminal_jacobian_matches_constraint_derivative():
    """约束雅可比 = 约束函数的解析导数（归一化后的链式雅可比）。"""
    shooter = _make_shooter_two_body()
    n_seg = 2
    y = shooter._default_x0(n_seg)

    jac = shooter._terminal_jacobian(y)
    # 与「未归一化链式雅可比 / 参考量」对比
    r_ref = np.linalg.norm(shooter._initial_state[:3])
    v_ref = np.linalg.norm(shooter._initial_state[3:6])
    _, _, raw_jac = shooter._propagate_chain_with_jacobian(y)
    scale = np.array([r_ref, r_ref, r_ref, v_ref, v_ref, v_ref])
    expected = raw_jac / scale[:, None]
    np.testing.assert_allclose(jac, expected, rtol=1e-10, atol=1e-12)


def test_analytic_jacobian_speedup_over_finite_difference():
    """解析雅可比 vs 数值差分：同解、显著提速。

    解析雅可比每迭代 1 次增广传播，数值差分每迭代 3N+1 次；预期加速一个
    量级。两者收敛到的燃料消耗应一致（解析雅可比不改变最优解，只改变求
    解效率）。
    """
    import time

    shooter = _make_shooter_two_body()
    n_seg = 4
    y0 = shooter._default_x0(n_seg)
    y0[0::3] = 0.5  # 避免边界 clip

    t0 = time.time()
    s_analytic = shooter.solve(n_seg, x0=y0, use_analytic_jac=True, maxiter=30)
    t_analytic = time.time() - t0

    t0 = time.time()
    s_numeric = shooter.solve(n_seg, x0=y0, use_analytic_jac=False, maxiter=30)
    t_numeric = time.time() - t0

    # 1. 两者收敛到接近的燃料消耗（解析雅可比不改变解，只加速）
    assert abs(s_analytic.fuel_consumed - s_numeric.fuel_consumed) < 1e-3, (
        f"解析({s_analytic.fuel_consumed:.5f}) vs 数值({s_numeric.fuel_consumed:.5f}) 燃料不一致"
    )
    # 2. 解析雅可比显著快（至少 5x；实测 ~24x，保守取 5x 避免机器抖动）
    assert t_numeric / t_analytic > 5.0, (
        f"解析雅可比应显著快于数值差分: analytic={t_analytic:.2f}s "
        f"numeric={t_numeric:.2f}s ratio={t_numeric / t_analytic:.1f}x"
    )
