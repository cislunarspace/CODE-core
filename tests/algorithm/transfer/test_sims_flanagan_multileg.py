"""Sims-Flanagan 多 leg + flyby 节点与成本函数集验证（issue #741）。

oracle 口径（ADR 0055）：④ 机制（同一转录既是生成器又是模型：合成可行链在
真值处等式残差为传播噪声级；解析雅可比对中心差分逐列对照，覆盖冲量、v∞ 锚、
TOF 三类变量与四种成本）、② 不变量（flyby 等模与近心点下界、跨 leg 全局时序
质量链段界）、③ 退化（弹道多交会链零解、r_p_min → 0 无约束极限、单 leg 形状
与 MVP 同输入一致）、① 闭式（flyby_turn_angle 与解报告的转角/近心点互逆）。
⑦ MALTO（AIAA 2006-6746）目标集合枚举仅作人工对照出处且原文未取得；本仓
GTOC 口径为自定义定义性公式（模块 docstring），无结果性数值断言。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.transfer import (
    G0_MPS2,
    SimsFlanaganMultiLegProblem,
    SimsFlanaganNode,
    SimsFlanaganProblem,
    SimsFlanaganPropulsion,
    flyby_turn_angle,
)
from e2m2e.integrators import propagate_kepler_py

pytestmark = [pytest.mark.orchestration, pytest.mark.low_thrust]

if propagate_kepler_py is None:
    pytest.skip("propagate_kepler_py 需要 make dev 构建", allow_module_level=True)

MU = 398600.435507  # 地球 GM（km³/s²），固定值便于复算
R0 = 7000.0  # km，圆轨道半径
V0 = float(np.sqrt(MU / R0))
DEP = np.array([R0, 0.0, 0.0, 0.0, V0, 0.0])
PERIOD = 2.0 * float(np.pi) * float(np.sqrt(R0**3 / MU))
C_KMS = 3000.0 * G0_MPS2 / 1000.0  # Isp=3000 s 的排气速度（km/s）

# 合成链参数（机制/不变量/FD 共用同一拓扑）：
_N_SEG = 6
_DT = 1500.0
_W0 = np.tile(np.array([0.0, 1.5e-3, 0.0]), (_N_SEG, 1))  # leg0 合成冲量
_W1 = np.tile(np.array([1.0e-3, 0.0, 0.5e-3]), (_N_SEG, 1))  # leg1 合成冲量
_V_IN = 2.0 * np.array([0.0, 0.0, 1.0])
_V_OUT = 2.0 * np.array([np.sin(np.radians(30.0)), 0.0, np.cos(np.radians(30.0))])
# r_p_min=250000 km、v_eff≈2 km/s → δ_max≈33.2° > 合成转角 30°：可行且非退化。
_R_P_MIN = 250000.0


def _kepler(x: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(
        propagate_kepler_py(np.asarray(x, dtype=float).tolist(), [t], MU)["states"][0],
        dtype=float,
    )


def _segmented_propagate(departure: np.ndarray, impulses: np.ndarray, dt: float) -> np.ndarray:
    """与 Sims-Flanagan 转录同构的段中冲量前向传播（合成控制的生成器）。"""
    x = np.asarray(departure, dtype=float).copy()
    half = 0.5 * dt
    for dv_k in np.asarray(impulses, dtype=float):
        mid = _kepler(x, half)
        mid[3:] += dv_k
        x = _kepler(mid, half)
    return x


def _flyby_chain(
    r_p_min_km: float = _R_P_MIN, thrust_n: float = 5.0
) -> SimsFlanaganMultiLegProblem:
    """合成可行 2 leg + 1 flyby 链（转角 30° < δ_max）。

    leg0 自 DEP 以 _W0 传播到 ``[r_B; v_B + v∞_in]``，leg1 自
    ``[r_B; v_B + v∞_out]`` 以 _W1 传播到终点交会态；真值决策向量由
    :func:`_flyby_chain_truth` 按同一布局拼接。默认初猜收敛需要高推力档
    （5000 N），真值初猜在 5 N 档即可收敛。
    """
    a0 = _segmented_propagate(DEP, _W0, _DT)
    v_body = a0[3:] - _V_IN
    start1 = np.concatenate([a0[:3], v_body + _V_OUT])
    final = _segmented_propagate(start1, _W1, _DT)
    flyby = SimsFlanaganNode(
        "flyby", np.concatenate([a0[:3], v_body]), mu_km3_s2=MU, r_p_min_km=r_p_min_km
    )
    end = SimsFlanaganNode("rendezvous", final)
    return SimsFlanaganMultiLegProblem(
        DEP,
        [flyby, end],
        [_N_SEG * _DT, _N_SEG * _DT],
        SimsFlanaganPropulsion.constant(thrust_n, 3000.0),
        1000.0,
        MU,
        backend="conic",
    )


def _flyby_chain_truth(problem: SimsFlanaganMultiLegProblem) -> np.ndarray:
    """合成链真值决策向量 [ΔV leg0, ΔV leg1, v∞_in, v∞_out]（TOF 固定档）。"""
    layout = problem._make_layout([_N_SEG, _N_SEG], "min_fuel", None, tof_free=False)
    return np.concatenate([_W0.reshape(-1), _W1.reshape(-1), _V_IN, _V_OUT])[: layout.n_var]


def _ml_segment_caps_km_s(
    sol, prop: SimsFlanaganPropulsion, m0_kg: float, *, reset_each_leg: bool = False
) -> np.ndarray:
    """从多 leg 解的字段精确复算各段 cap（全局时序质量链，跨 leg 版 MVP helper）。

    ``reset_each_leg=True`` 时每 leg 质量链重置回 m₀（对照组：证明求解器
    实际执行的是全局时序链，早 leg 耗燃后晚 leg 段界更紧）。
    """
    impulses = [np.asarray(leg.impulses_km_s, dtype=float) for leg in sol.legs]
    norms = [np.linalg.norm(w, axis=1) for w in impulses]
    flat = np.concatenate(norms)
    if reset_each_leg:
        prefix = np.concatenate([np.concatenate([[0.0], np.cumsum(x)[:-1]]) for x in norms])
    else:
        prefix = np.concatenate([[0.0], np.cumsum(flat)[:-1]])
    m_bar = m0_kg * np.exp(-prefix / C_KMS)
    caps = []
    idx = 0
    for leg, w in enumerate(impulses):
        n = w.shape[0]
        m = n // 2
        dt = float(sol.leg_tofs_s[leg]) / n
        for k in range(n):
            if k < m:
                anchor = np.asarray(sol.legs[leg].forward_states[k], dtype=float)
                half = 0.5 * dt
            else:
                anchor = np.asarray(sol.legs[leg].backward_states[k - m + 1], dtype=float)
                half = -0.5 * dt
            mid = _kepler(anchor, half)
            thrust = prop.max_thrust_n(float(np.linalg.norm(mid[:3])))
            caps.append(thrust / m_bar[idx] * dt / 1000.0)
            idx += 1
    return np.asarray(caps)


def _ballistic_chain() -> SimsFlanaganMultiLegProblem:
    """同一圆轨道弧上的弹道 2 leg 多交会链（节点取自 T/4、T/2 传播）。"""
    node0 = SimsFlanaganNode("rendezvous", _kepler(DEP, PERIOD / 4))
    node1 = SimsFlanaganNode("rendezvous", _kepler(DEP, PERIOD / 2))
    return SimsFlanaganMultiLegProblem(
        DEP,
        [node0, node1],
        [PERIOD / 4, PERIOD / 4],
        SimsFlanaganPropulsion.constant(5000.0, 3000.0),
        1000.0,
        MU,
        backend="conic",
    )


def test_multileg_input_validation():
    """入参与配置校验（接口行为，无 oracle 数值）。"""
    prop = SimsFlanaganPropulsion.constant(5.0, 3000.0)
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    node = SimsFlanaganNode("rendezvous", state)
    args = (DEP, [node], [6000.0], prop, 1000.0, MU)

    # backend 必填关键字：缺失即 Python 原生 TypeError；ephemeris 档上下文校验。
    with pytest.raises(TypeError):
        SimsFlanaganMultiLegProblem(*args)
    with pytest.raises(ValueError, match="必须提供 epoch_et_s"):
        SimsFlanaganMultiLegProblem(*args, backend="ephemeris")
    with pytest.raises(ValueError, match="必须提供 ephemeris_system"):
        SimsFlanaganMultiLegProblem(*args, backend="ephemeris", epoch_et_s=0.0)
    with pytest.raises(ValueError, match="不接受"):
        SimsFlanaganMultiLegProblem(*args, backend="conic", epoch_et_s=0.0)
    with pytest.raises(ValueError, match="必须为"):
        SimsFlanaganMultiLegProblem(*args, backend="bogus")
    # 节点类型。
    with pytest.raises(ValueError, match="kind"):
        SimsFlanaganNode("bogus", state)
    with pytest.raises(ValueError, match="state"):
        SimsFlanaganNode("rendezvous", state[:4])
    with pytest.raises(ValueError, match="有限"):
        SimsFlanaganNode("rendezvous", np.append(state[:5], np.inf))
    with pytest.raises(ValueError, match="必须提供"):
        SimsFlanaganNode("flyby", state)  # 缺 mu/r_p_min
    with pytest.raises(ValueError, match="必须提供"):
        SimsFlanaganNode("flyby", state, mu_km3_s2=MU)
    with pytest.raises(ValueError, match="正的有限"):
        SimsFlanaganNode("flyby", state, mu_km3_s2=MU, r_p_min_km=0.0)
    with pytest.raises(ValueError, match="不接受"):
        SimsFlanaganNode("rendezvous", state, mu_km3_s2=MU)
    # 链结构与 leg_tofs。
    fly_end = SimsFlanaganNode("flyby", state, mu_km3_s2=MU, r_p_min_km=6500.0)
    with pytest.raises(ValueError, match="不能为空"):
        SimsFlanaganMultiLegProblem(DEP, [], [], prop, 1000.0, MU, backend="conic")
    with pytest.raises(ValueError, match="末节点"):
        SimsFlanaganMultiLegProblem(DEP, [fly_end], [6000.0], prop, 1000.0, MU, backend="conic")
    with pytest.raises(ValueError, match="leg_tofs_s"):
        SimsFlanaganMultiLegProblem(
            DEP, [node], [6000.0, 6000.0], prop, 1000.0, MU, backend="conic"
        )
    with pytest.raises(ValueError, match="正的有限"):
        SimsFlanaganMultiLegProblem(DEP, [node], [0.0], prop, 1000.0, MU, backend="conic")
    with pytest.raises(ValueError, match="nodes\\[0\\]"):
        SimsFlanaganMultiLegProblem(DEP, [state], [6000.0], prop, 1000.0, MU, backend="conic")
    # solve 参数。
    problem = SimsFlanaganMultiLegProblem(*args, backend="conic")
    with pytest.raises(ValueError, match="n_segments"):
        problem.solve([2, 2])  # 序列长度 ≠ 节点数 1
    with pytest.raises(ValueError, match="n_segments"):
        problem.solve(1)
    with pytest.raises(ValueError, match="n_segments"):
        problem.solve([2.5])
    with pytest.raises(ValueError, match="cost"):
        problem.solve(4, cost="bogus")
    with pytest.raises(ValueError, match="cost_weights"):
        problem.solve(4, cost="min_time", tof_bounds_s=(100.0, 200.0), cost_weights=(1.0, 1.0))
    with pytest.raises(ValueError, match="cost_weights"):
        problem.solve(4, cost="weighted", tof_bounds_s=(100.0, 200.0))
    with pytest.raises(ValueError, match="cost_weights"):
        problem.solve(4, cost="weighted", tof_bounds_s=(100.0, 200.0), cost_weights=(1.0, 0.0))
    with pytest.raises(ValueError, match="cost_weights"):
        problem.solve(4, cost="weighted", tof_bounds_s=(100.0, 200.0), cost_weights=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="tof_bounds_s"):
        problem.solve(4, cost="min_time")
    with pytest.raises(ValueError, match="tof_bounds_s"):
        problem.solve(4, cost="weighted", cost_weights=(1.0, 1.0))
    with pytest.raises(ValueError, match="tof_bounds_s"):
        problem.solve(4, tof_bounds_s=(0.0, 200.0))
    with pytest.raises(ValueError, match="tof_bounds_s"):
        problem.solve(4, tof_bounds_s=(300.0, 200.0))
    with pytest.raises(ValueError, match="tof_bounds_s"):
        problem.solve(4, tof_bounds_s=(100.0, 200.0, 300.0))
    with pytest.raises(ValueError, match="x0"):
        problem.solve(4, x0=np.zeros(5))
    with pytest.raises(ValueError, match="guess"):
        problem.solve(4, guess="bogus")


def test_mechanism_synthetic_flyby_chain():
    """合成可行链的转录复现（④ 机制：同一转录既是生成器又是模型）。

    直接评估证明真值决策向量处等式残差为闭式 Kepler 传播噪声级、flyby
    转角裕度非负；从真值出发求解收敛、残差 ≈ 0 且燃料不劣于合成控制。
    可行流形为 5 维（18 变量 − 13 等式），SLSQP 会沿流形降燃料，故不复制
    MVP 孤立可行点用例的 nit ≤ 3 断言。
    """
    problem = _flyby_chain()
    x_true = _flyby_chain_truth(problem)
    layout = problem._make_layout([_N_SEG, _N_SEG], "min_fuel", None, tof_free=False)
    ev = problem._evaluate(x_true, layout, with_sens=False)
    assert float(np.max(np.abs(ev.eq))) < 1e-12
    assert float(np.min(ev.ineq)) >= 0.0  # 段界 + flyby 转角全可行
    assert ev.flybys[0].delta == pytest.approx(np.radians(30.0), abs=1e-12)

    sol = problem.solve(_N_SEG, x0=x_true)
    assert sol.status.name == "CONVERGED", sol.message
    synthetic_total = float(np.sum(np.linalg.norm(_W0, axis=1) + np.linalg.norm(_W1, axis=1)))
    assert sol.delta_v_total_km_s <= synthetic_total + 1e-9
    for leg in sol.legs:
        assert float(np.max(np.abs(leg.matchpoint_residual[:3]))) < 1e-5  # km
        assert float(np.max(np.abs(leg.matchpoint_residual[3:]))) < 1e-6  # km/s
    # 火箭方程恒等与成本回显。
    all_impulses = np.vstack([lg.impulses_km_s for lg in sol.legs])
    total_dv = float(np.sum(np.linalg.norm(all_impulses, axis=1)))
    assert sol.delta_v_total_km_s == pytest.approx(total_dv, rel=1e-12)
    assert sol.final_mass_kg == pytest.approx(1000.0 * float(np.exp(-total_dv / C_KMS)), abs=1e-9)
    assert sol.cost == "min_fuel"
    assert sol.objective_value == pytest.approx(sol.delta_v_total_km_s, abs=1e-6)


def test_ballistic_chain_and_single_leg_equivalence():
    """③ 退化 + 跨类一致：弹道链零解；单 leg 形状 ≡ MVP 同输入。"""
    # 弹道多交会链：x0=0 已是全局最优，一步收敛、ΔV ≈ 0。
    chain = _ballistic_chain()
    sol = chain.solve(6)
    assert sol.status.name == "CONVERGED", sol.message
    assert sol.n_iter <= 3
    assert sol.delta_v_total_km_s < 1e-8
    assert abs(sol.final_mass_kg - 1000.0) <= 1e-12
    for leg in sol.legs:
        assert float(np.max(np.abs(leg.matchpoint_residual))) < 1e-8

    # 单 leg 形状的多 leg 问题与 MVP 同输入同段数同初猜 → 同一 NLP。
    n, dt = 6, 1500.0
    w = np.tile(np.array([0.0, 2.0e-3, 0.0]), (n, 1))
    arrival = _segmented_propagate(DEP, w, dt)
    prop = SimsFlanaganPropulsion.constant(5.0, 3000.0)
    mvp = SimsFlanaganProblem(DEP, arrival, n * dt, prop, 1000.0, MU, backend="conic")
    single = SimsFlanaganMultiLegProblem(
        DEP,
        [SimsFlanaganNode("rendezvous", arrival)],
        [n * dt],
        prop,
        1000.0,
        MU,
        backend="conic",
    )
    sol_mvp = mvp.solve(n, guess="edelbaum")
    sol_ml = single.solve(n, guess="edelbaum")
    assert sol_mvp.status.name == "CONVERGED", sol_mvp.message
    assert sol_ml.status.name == "CONVERGED", sol_ml.message
    assert sol_ml.delta_v_total_km_s == pytest.approx(sol_mvp.delta_v_total_km_s, abs=1e-6)


def test_flyby_invariants_and_global_mass_chain():
    """② 不变量：flyby 等模、近心点下界、跨 leg 全局时序质量链段界。"""
    problem = _flyby_chain(thrust_n=5000.0)
    sol = problem.solve(_N_SEG, maxiter=400)
    assert sol.status.name == "CONVERGED", sol.message
    assert len(sol.flybys) == 1
    fb = sol.flybys[0]
    n_in = float(np.linalg.norm(fb.v_inf_in_km_s))
    n_out = float(np.linalg.norm(fb.v_inf_out_km_s))
    assert abs(n_in - n_out) < 1e-6  # km/s
    assert fb.v_inf_km_s == pytest.approx(n_in, rel=1e-12)
    # 转角不等式 ⇔ 近心点下界（同一闭式的单调重排）。
    assert fb.pericenter_radius_km >= _R_P_MIN - 1e-6
    assert fb.turn_angle_rad < float(np.pi)

    # 全局时序质量链：解满足全局链段界；且晚 leg 的 cap 按 ``T/m̄·dt`` 精确
    # 承接早 leg 耗燃——m̄ = m₀·exp(−p/c) 随全局前缀耗损 → cap 比每 leg 质量
    # 重置的对照放松 exp(p_early/c) 因子（耗燃使加速度上升），差值恰为火箭
    # 方程因子，证明质量链跨 leg 传递而非逐 leg 重置。
    prop = problem._propulsion
    norms = np.linalg.norm(
        np.vstack([np.asarray(lg.impulses_km_s, dtype=float) for lg in sol.legs]), axis=1
    )
    caps_global = _ml_segment_caps_km_s(sol, prop, 1000.0)
    caps_reset = _ml_segment_caps_km_s(sol, prop, 1000.0, reset_each_leg=True)
    assert float(np.max(norms / caps_global)) <= 1.0 + 1e-9
    n_seg_leg0 = _N_SEG
    early_dv = float(np.sum(norms[:n_seg_leg0]))
    assert early_dv > 1e-3  # 早 leg 确有耗燃（否则对照无分辨力）
    loosen = float(np.min(caps_global[n_seg_leg0:] / caps_reset[n_seg_leg0:]))
    assert loosen == pytest.approx(float(np.exp(early_dv / C_KMS)), rel=1e-6)
    assert loosen > 1.0 + 1e-5


def test_analytic_jacobian_matches_finite_difference():
    """④ 机制：eq/ineq/目标解析雅可比对中心差分逐列对照（冲量、v∞、TOF 三类）。"""
    n, dt = 4, 1500.0
    w0 = np.tile(np.array([1.0e-3, -0.8e-3, 0.5e-3]), (n, 1))
    w1 = np.tile(np.array([0.6e-3, 0.3e-3, -0.4e-3]), (n, 1))
    a0 = _segmented_propagate(DEP, w0, dt)
    v_body = a0[3:] - _V_IN
    start1 = np.concatenate([a0[:3], v_body + _V_OUT])
    final = _segmented_propagate(start1, w1, dt)
    problem = SimsFlanaganMultiLegProblem(
        DEP,
        [
            SimsFlanaganNode(
                "flyby",
                np.concatenate([a0[:3], v_body]),
                mu_km3_s2=MU,
                r_p_min_km=_R_P_MIN,
            ),
            SimsFlanaganNode("rendezvous", final),
        ],
        [n * dt, n * dt],
        SimsFlanaganPropulsion.constant(5.0, 3000.0),
        1000.0,
        MU,
        backend="conic",
    )
    x_test = np.concatenate(
        [w0.reshape(-1), w1.reshape(-1), _V_IN, _V_OUT, [n * dt * 0.97, n * dt * 1.03]]
    )
    cases = [
        ("min_fuel", None),
        ("min_time", None),
        ("weighted", (1.0, 2.0)),
        ("gtoc", None),
    ]
    for cost, weights in cases:
        layout = problem._make_layout([n, n], cost, weights, tof_free=True)
        ev = problem._evaluate(x_test, layout, with_sens=True)
        assert ev.eq_jac is not None and ev.ineq_jac is not None and ev.obj_grad is not None
        for i in range(layout.n_var):
            h = 1e-2 if i >= layout.tof_off else 1e-6
            x_plus = x_test.copy()
            x_plus[i] += h
            x_minus = x_test.copy()
            x_minus[i] -= h
            ev_plus = problem._evaluate(x_plus, layout, with_sens=False)
            ev_minus = problem._evaluate(x_minus, layout, with_sens=False)
            fd_eq = (ev_plus.eq - ev_minus.eq) / (2.0 * h)
            fd_ineq = (ev_plus.ineq - ev_minus.ineq) / (2.0 * h)
            fd_obj = (ev_plus.objective - ev_minus.objective) / (2.0 * h)
            np.testing.assert_allclose(fd_eq, ev.eq_jac[:, i], atol=1e-5)
            np.testing.assert_allclose(fd_ineq, ev.ineq_jac[:, i], atol=1e-4)
            assert fd_obj == pytest.approx(float(ev.obj_grad[i]), abs=1e-5)


def test_cost_functions():
    """成本函数集行为：min_time 缩短、weighted 单调、gtoc 与 min_fuel 同最优类。"""
    chain = _ballistic_chain()
    n = 6
    nominal = float(np.sum(chain._tof_nominal))
    lb, ub = 0.5 * PERIOD / 4, 1.5 * PERIOD / 4

    # min_time：高推力下总 TOF 显著小于名义（受段界约束的可行最短）。
    sol_t = chain.solve(n, cost="min_time", tof_bounds_s=(lb, ub))
    assert sol_t.status.name == "CONVERGED", sol_t.message
    assert float(np.sum(sol_t.leg_tofs_s)) < 0.9 * nominal
    assert sol_t.objective_value == pytest.approx(float(np.sum(sol_t.leg_tofs_s)) / nominal)

    # weighted：时间权重加重 → 总 TOF 单调变短。
    tof_light = chain.solve(n, cost="weighted", cost_weights=(1.0, 1e-3), tof_bounds_s=(lb, ub))
    tof_heavy = chain.solve(n, cost="weighted", cost_weights=(1.0, 50.0), tof_bounds_s=(lb, ub))
    assert tof_light.status.name == "CONVERGED", tof_light.message
    assert tof_heavy.status.name == "CONVERGED", tof_heavy.message
    assert float(np.sum(tof_heavy.leg_tofs_s)) < float(np.sum(tof_light.leg_tofs_s)) - 1.0

    # gtoc 单终点交会（L=1 形状）与 min_fuel 同 argmin：ΔV 总量一致。
    n1, dt1 = 6, 1500.0
    w = np.tile(np.array([0.0, 2.0e-3, 0.0]), (n1, 1))
    arrival = _segmented_propagate(DEP, w, dt1)
    prop = SimsFlanaganPropulsion.constant(5.0, 3000.0)
    single = SimsFlanaganMultiLegProblem(
        DEP,
        [SimsFlanaganNode("rendezvous", arrival)],
        [n1 * dt1],
        prop,
        1000.0,
        MU,
        backend="conic",
    )
    sol_g = single.solve(n1, cost="gtoc", guess="edelbaum")
    sol_f = single.solve(n1, guess="edelbaum")
    assert sol_g.status.name == "CONVERGED", sol_g.message
    assert sol_f.status.name == "CONVERGED", sol_f.message
    assert sol_g.delta_v_total_km_s == pytest.approx(sol_f.delta_v_total_km_s, abs=1e-6)
    # 单终点口径 J = −exp(−Σ‖ΔV‖/c)。
    assert sol_g.objective_value == pytest.approx(
        -float(np.exp(-sol_g.delta_v_total_km_s / C_KMS)), rel=1e-9
    )

    # gtoc 多交会（2 leg 全 rendezvous）：J 由解冲量按定义式复算相等。
    w0 = np.tile(np.array([0.0, 1.5e-3, 0.0]), (n1, 1))
    a0 = _segmented_propagate(DEP, w0, dt1)
    w1 = np.tile(np.array([1.0e-3, 0.0, 0.5e-3]), (n1, 1))
    final = _segmented_propagate(a0, w1, dt1)
    multi = SimsFlanaganMultiLegProblem(
        DEP,
        [SimsFlanaganNode("rendezvous", a0), SimsFlanaganNode("rendezvous", final)],
        [n1 * dt1, n1 * dt1],
        prop,
        1000.0,
        MU,
        backend="conic",
    )
    x_true = np.concatenate([w0.reshape(-1), w1.reshape(-1)])
    sol_m = multi.solve(n1, cost="gtoc", x0=x_true)
    assert sol_m.status.name == "CONVERGED", sol_m.message
    leg_sums = np.cumsum(
        [
            float(np.sum(np.linalg.norm(np.asarray(lg.impulses_km_s, dtype=float), axis=1)))
            for lg in sol_m.legs
        ]
    )
    j_recompute = -float(np.sum(np.exp(-leg_sums / C_KMS)))
    assert sol_m.objective_value == pytest.approx(j_recompute, rel=1e-9)


def test_flyby_degenerate_and_closed_form():
    """③/①：r_p_min → 0 无约束极限可收敛；转角/近心点与 mga 闭式互逆。"""
    # r_p_min → 0：δ_max → π（近无约束），真值初猜下收敛且转角自由实现。
    problem = _flyby_chain(r_p_min_km=1.0)
    sol = problem.solve(_N_SEG, x0=_flyby_chain_truth(problem))
    assert sol.status.name == "CONVERGED", sol.message
    assert sol.flybys[0].pericenter_radius_km >= 1.0 - 1e-6

    # ① 闭式互逆：解的 (δ, v_eff) 反解 r_p，再正算转角应回到 δ。
    problem = _flyby_chain()
    sol = problem.solve(_N_SEG, x0=_flyby_chain_truth(problem), maxiter=400)
    assert sol.status.name == "CONVERGED", sol.message
    fb = sol.flybys[0]
    assert fb.turn_angle_rad > 1e-3  # 非零转角（有限 r_p 的互逆才有意义）
    delta_back = flyby_turn_angle(fb.pericenter_radius_km, fb.v_inf_km_s, MU)
    assert delta_back == pytest.approx(fb.turn_angle_rad, abs=1e-9)
