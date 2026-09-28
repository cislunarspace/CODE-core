"""Sims-Flanagan 单 leg rendezvous 预设计求解器验证（issue #740）。

对照 issue #740 验收清单：后验闸门与入参校验（接口）、合成控制转录复现（④
机制）、匹配点连续性与段可行域（② 不变量）、零推力退化（③）、日心 Hohmann
ΔV 带（① 带）、解析雅可比对中心差分（④ 机制，守 #739 STM 推导链）。

oracle 口径（ADR 0055）：① 带下界为 Hohmann 双脉冲闭式、上界为 Edelbaum
闭式（① 定义性公式，容差 2% 离散裕度）；② 为不变量复算；③ 为退化对照；
④ 为机制/守恒与 FD 交叉验证。⑦ 类结果性数值（文献算例收敛 ΔV）禁入断言。

纯 conic 档二体，无需 SPICE 内核。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.transfer import (
    G0_MPS2,
    SimsFlanaganProblem,
    SimsFlanaganPropulsion,
    edelbaum_delta_v,
    hohmann_delta_v,
    hohmann_tof,
)
from e2m2e.integrators import propagate_kepler_py

pytestmark = [pytest.mark.orchestration, pytest.mark.low_thrust]

if propagate_kepler_py is None:
    pytest.skip("propagate_kepler_py 需要 make dev 构建", allow_module_level=True)

MU = 398600.435507  # 地球 GM（km³/s²），固定值便于复算
MU_SUN = 1.32712440018e11  # 太阳 GM（km³/s²），DE421 口径
AU_KM = 1.495978707e8  # IAU 2012 天文单位（km）


def _helio_150_problem(prop: SimsFlanaganPropulsion) -> SimsFlanaganProblem:
    """1 AU → 1.5 AU 共面圆-圆、180° Hohmann 定相、tof = Hohmann 半周期的日心问题。"""
    r1, r2 = AU_KM, 1.5 * AU_KM
    v1, v2 = float(np.sqrt(MU_SUN / r1)), float(np.sqrt(MU_SUN / r2))
    departure = np.array([r1, 0.0, 0.0, 0.0, v1, 0.0])
    arrival = np.array([-r2, 0.0, 0.0, 0.0, -v2, 0.0])  # 顺行：θ=π 处速度为 −v₂·ŷ
    tof = hohmann_tof(r1, r2, MU_SUN)
    return SimsFlanaganProblem(departure, arrival, tof, prop, 1000.0, MU_SUN, backend="conic")


def _segmented_propagate(
    departure: np.ndarray, impulses: np.ndarray, dt: float, mu: float
) -> np.ndarray:
    """与 Sims-Flanagan 转录同构的段中冲量前向传播（合成控制的生成器）。"""
    x = np.asarray(departure, dtype=float).copy()
    half = 0.5 * dt
    for dv_k in np.asarray(impulses, dtype=float):
        mid = np.asarray(propagate_kepler_py(x.tolist(), [half], mu)["states"][0], dtype=float)
        mid[3:] += dv_k
        x = np.asarray(propagate_kepler_py(mid.tolist(), [half], mu)["states"][0], dtype=float)
    return x


def _segment_caps_km_s(sol, prop: SimsFlanaganPropulsion, m0_kg: float, mu: float) -> np.ndarray:
    """从解的暴露字段精确复算各段 cap_k = T(r̄ₖ)/m̄ₖ·Δt（N/kg·s → km/s）。

    r̄ₖ 取求解器同一来源：前向段自 forward_states[k] 顺推半段、后向段自
    backward_states[k-m+1] 逆推半段；m̄ₖ = m₀·exp(−Σ_{j<k}‖ΔVⱼ‖/c)。
    """
    impulses = np.asarray(sol.impulses_km_s, dtype=float)
    n = impulses.shape[0]
    m = n // 2
    dt = float(sol.node_times_s[1] - sol.node_times_s[0])
    c_kms = prop.isp_s * G0_MPS2 / 1000.0
    norms = np.linalg.norm(impulses, axis=1)
    prefix = np.concatenate([[0.0], np.cumsum(norms)[:-1]])
    m_bar = m0_kg * np.exp(-prefix / c_kms)
    caps = np.zeros(n)
    for k in range(n):
        if k < m:
            anchor = np.asarray(sol.forward_states[k], dtype=float)
            half = 0.5 * dt
        else:
            anchor = np.asarray(sol.backward_states[k - m + 1], dtype=float)
            half = -0.5 * dt
        mid = np.asarray(propagate_kepler_py(anchor.tolist(), [half], mu)["states"][0], dtype=float)
        caps[k] = prop.max_thrust_n(float(np.linalg.norm(mid[:3]))) / m_bar[k] * dt / 1000.0
    return caps


def test_backend_and_input_validation():
    """backend 必填与入参/配置校验（接口行为，无 oracle 数值）。"""
    departure = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    arrival = departure.copy()
    prop = SimsFlanaganPropulsion.constant(5.0, 3000.0)

    # backend 为必填关键字：缺失即 Python 原生 TypeError（缺失显式报错）。
    with pytest.raises(TypeError):
        SimsFlanaganProblem(departure, arrival, 6000.0, prop, 1000.0, MU)
    # ephemeris 档未实现：显式报错，不静默回退（ADR 0050 理由 6）。
    with pytest.raises(ValueError, match="未实现"):
        SimsFlanaganProblem(departure, arrival, 6000.0, prop, 1000.0, MU, backend="ephemeris")
    # 非法档位名。
    with pytest.raises(ValueError, match="必须为"):
        SimsFlanaganProblem(departure, arrival, 6000.0, prop, 1000.0, MU, backend="bogus")
    # 状态维度、tof、质量、mu。
    with pytest.raises(ValueError, match="departure_state"):
        SimsFlanaganProblem(departure[:5], arrival, 6000.0, prop, 1000.0, MU, backend="conic")
    with pytest.raises(ValueError, match="arrival_state"):
        SimsFlanaganProblem(
            departure, np.asarray(arrival)[:3], 6000.0, prop, 1000.0, MU, backend="conic"
        )
    with pytest.raises(ValueError, match="tof_s"):
        SimsFlanaganProblem(departure, arrival, 0.0, prop, 1000.0, MU, backend="conic")
    with pytest.raises(ValueError, match="initial_mass_kg"):
        SimsFlanaganProblem(departure, arrival, 6000.0, prop, 0.0, MU, backend="conic")
    with pytest.raises(ValueError, match="mu_km3_s2"):
        SimsFlanaganProblem(departure, arrival, 6000.0, prop, 1000.0, 0.0, backend="conic")
    # 段数下界。
    problem = SimsFlanaganProblem(departure, arrival, 6000.0, prop, 1000.0, MU, backend="conic")
    with pytest.raises(ValueError, match="n_segments"):
        problem.solve(1)
    # 推进配置：双模式同填 / 全空 / 非法效率。
    with pytest.raises(ValueError, match="恰好指定一个"):
        SimsFlanaganPropulsion(isp_s=3000.0, t_max_n=5.0, p0_w=1e4).validate()
    with pytest.raises(ValueError, match="恰好指定一个"):
        SimsFlanaganPropulsion(isp_s=3000.0).validate()
    with pytest.raises(ValueError, match="efficiency"):
        SimsFlanaganPropulsion(isp_s=3000.0, p0_w=1e4, efficiency=0.0).validate()
    with pytest.raises(ValueError, match="x0"):
        problem.solve(4, x0=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="guess"):
        problem.solve(4, guess="bogus")


def test_mechanism_known_control_reproduction():
    """合成冲量控制的转录复现（④ 机制：同一转录既是生成器又是模型）。

    n=2：6 维等式约束对 6 维决策变量在合成点附近非奇异，可行集退化为孤立
    点——求解器必须逐位收回合成控制（nit ≤ 3、末质量火箭方程闭式）。
    n=6：合成控制只是可行点而非最优，SLSQP 应收敛到更低燃料的可行解；
    断言收敛、末质量恒等式与“不劣于合成控制”。
    """
    r0 = 7000.0
    v0 = float(np.sqrt(MU / r0))
    departure = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
    prop = SimsFlanaganPropulsion.constant(5.0, 3000.0)
    c_kms = 3000.0 * G0_MPS2 / 1000.0
    m0 = 1000.0

    # -- n=2：孤立可行点的精确复现 --
    n, dt = 2, 2000.0
    x0 = np.tile(np.array([0.0, 2e-3, 0.0]), (n, 1))  # cap = 5e-3 km/s，可行
    arrival = _segmented_propagate(departure, x0, dt, MU)
    problem = SimsFlanaganProblem(departure, arrival, n * dt, prop, m0, MU, backend="conic")
    sol = problem.solve(n, x0=x0)
    assert sol.status.name == "CONVERGED", sol.message
    assert sol.n_iter <= 3
    np.testing.assert_allclose(sol.impulses_km_s, x0, atol=1e-5)
    expected_mf = m0 * float(np.exp(-2.0 * 2e-3 / c_kms))
    assert sol.final_mass_kg == pytest.approx(expected_mf, abs=1e-9)
    assert float(np.max(np.abs(sol.matchpoint_residual))) < 1e-9

    # -- n=6：合成控制为可行初值，解应收敛且燃料不劣于合成控制 --
    n, dt = 6, 1000.0
    x0 = np.tile(np.array([0.0, 2e-3, 0.0]), (n, 1))
    arrival = _segmented_propagate(departure, x0, dt, MU)
    problem = SimsFlanaganProblem(departure, arrival, n * dt, prop, m0, MU, backend="conic")
    sol = problem.solve(n, x0=x0)
    assert sol.status.name == "CONVERGED", sol.message
    # 匹配点残差（转录对目标态闭合）。
    residual = np.asarray(sol.matchpoint_residual)
    assert float(np.max(np.abs(residual[:3]))) < 1e-6  # 位置 km
    assert float(np.max(np.abs(residual[3:]))) < 1e-9  # 速度 km/s
    # 火箭方程恒等：末质量由解自身的冲量范数决定。
    total_dv = float(np.sum(np.linalg.norm(sol.impulses_km_s, axis=1)))
    assert sol.delta_v_total_km_s == pytest.approx(total_dv, rel=1e-12)
    assert sol.final_mass_kg == pytest.approx(m0 * float(np.exp(-total_dv / c_kms)), abs=1e-9)
    assert sol.fuel_kg == pytest.approx(m0 - sol.final_mass_kg, abs=1e-9)
    # min-fuel 不劣于均匀合成控制（0.012 km/s）。
    assert sol.delta_v_total_km_s <= 0.012 + 1e-9
    # 段可行域。
    caps = _segment_caps_km_s(sol, prop, m0, MU)
    assert float(np.max(np.linalg.norm(sol.impulses_km_s, axis=1) / caps)) <= 1.0 + 1e-9


def test_matchpoint_continuity_invariant():
    """匹配点连续性：两 pass 状态一致、匹配点质量为恒等式（② 不变量）。

    n=6 交会解的前向/后向 pass 在匹配节点的位置/速度差为 SLSQP 收敛残差；
    质量维由 m̄ₖ = m₀·exp(−Σ_{j<k}‖ΔVⱼ‖/c) 的同一公式保证连续（恒等式）。
    """
    r0 = 7000.0
    v0 = float(np.sqrt(MU / r0))
    departure = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
    n, dt = 6, 1000.0
    x0 = np.tile(np.array([0.0, 2e-3, 0.0]), (n, 1))
    arrival = _segmented_propagate(departure, x0, dt, MU)
    prop = SimsFlanaganPropulsion.constant(5.0, 3000.0)
    problem = SimsFlanaganProblem(departure, arrival, n * dt, prop, 1000.0, MU, backend="conic")
    sol = problem.solve(n, x0=x0)
    assert sol.status.name == "CONVERGED", sol.message

    fwd_match = np.asarray(sol.forward_states[-1], dtype=float)
    bwd_match = np.asarray(sol.backward_states[0], dtype=float)
    assert float(np.max(np.abs(fwd_match[:3] - bwd_match[:3]))) < 1e-6  # km
    assert float(np.max(np.abs(fwd_match[3:] - bwd_match[3:]))) < 1e-9  # km/s

    # 匹配点质量恒等式：前向 m₀·exp(−Σ_{j<m}/c) ≡ 末质量·exp(Σ_{j≥m}/c)。
    c_kms = 3000.0 * G0_MPS2 / 1000.0
    norms = np.linalg.norm(sol.impulses_km_s, axis=1)
    m_match_fwd = 1000.0 * float(np.exp(-float(np.sum(norms[:3])) / c_kms))
    m_match_bwd = sol.final_mass_kg * float(np.exp(float(np.sum(norms[3:])) / c_kms))
    assert abs(m_match_fwd - m_match_bwd) < 1e-12


def test_segment_feasibility_bound():
    """段可行域 |ΔVₖ| ≤ capₖ 双模式验证（② 不变量）。

    常推力 0.5 N：段界全激活的推力饥饿解，压力测试 SLSQP 激活约束的
    贴合精度；SEP 7.4e4 W（T(1AU)≈5 N）：段界不激活的普通解，另断言
    推力随日心距平方反比衰减（功率衰减分支被走到）。
    """
    for prop in (
        SimsFlanaganPropulsion.constant(0.5, 3000.0),
        SimsFlanaganPropulsion.sep(7.4e4, 3000.0),
    ):
        problem = _helio_150_problem(prop)
        sol = problem.solve(16, guess="edelbaum")
        assert sol.status.name == "CONVERGED", sol.message
        caps = _segment_caps_km_s(sol, prop, 1000.0, MU_SUN)
        norms = np.linalg.norm(sol.impulses_km_s, axis=1)
        assert float(np.max(norms / caps)) <= 1.0 + 1e-9

    # SEP 功率衰减分支：T(2 AU) < T(1 AU)。
    sep_prop = SimsFlanaganPropulsion.sep(7.4e4, 3000.0)
    assert sep_prop.max_thrust_n(2.0 * AU_KM) < sep_prop.max_thrust_n(AU_KM)


def test_zero_thrust_degeneration():
    """零推力退化：arrival=departure、tof=整周期、x0=0 → 零解（③ 退化）。"""
    r0 = 7000.0
    tof = 2.0 * float(np.pi) * float(np.sqrt(r0**3 / MU))
    departure = np.array([r0, 0.0, 0.0, 0.0, float(np.sqrt(MU / r0)), 0.0])
    prop = SimsFlanaganPropulsion.constant(0.5, 3000.0)
    problem = SimsFlanaganProblem(
        departure, departure.copy(), tof, prop, 1000.0, MU, backend="conic"
    )
    sol = problem.solve(4, x0=np.zeros((4, 3)))
    assert sol.status.name == "CONVERGED", sol.message
    assert sol.delta_v_total_km_s < 1e-8
    assert abs(sol.final_mass_kg - 1000.0) <= 1e-12
    # 前向剖面与直接 Kepler 传播对照。闭式内核单 hop 数值噪声 ~1e-11 km
    # （实测 1.4e-11），半段接龙组合噪声同量级，1e-9 给两个量级裕度。
    half_tof = 0.5 * tof
    direct = np.asarray(
        propagate_kepler_py(departure.tolist(), [half_tof], MU)["states"][0], dtype=float
    )
    np.testing.assert_allclose(sol.forward_states[-1], direct, atol=1e-9)


def test_delta_v_sanity_band():
    """日心 1→1.5 AU 的 ΔV 带宽（① 带：Hohmann 下界、Edelbaum 上界）。

    ΔV_SF ≥ dv1+dv2（双脉冲 Hohmann 是同 tof 任务的冲量下界）；有限推力
    离散解不超过 Edelbaum 螺旋闭式的 1.02 倍。推力取 5 N（段界远不激活、
    近脉冲品质）：0.5 N 档在 Hohmann tof 内属推力饥饿工况（离散可达域
    收缩，总 ΔV 收敛到 ~2×Edelbaum，见 n=64/128 实测趋势），带宽断言
    不成立于该参数化，故本用例取近脉冲推力档。细化对照取 n=16→32：n=8
    解（5.9182）已超上界带宽（近脉冲档下 n=16 才进入带宽），依计划
    contingency 记录离散化理由；细化不减质断言（n=32 对照）保持。
    """
    r1, r2 = AU_KM, 1.5 * AU_KM
    v1 = float(np.sqrt(MU_SUN / r1))
    v2 = float(np.sqrt(MU_SUN / r2))
    dv1, dv2 = hohmann_delta_v(r1, r2, MU_SUN)
    edelbaum = edelbaum_delta_v(v1, v2)

    problem = _helio_150_problem(SimsFlanaganPropulsion.constant(5.0, 3000.0))
    sol16 = problem.solve(16, guess="edelbaum")
    assert sol16.status.name == "CONVERGED", sol16.message
    assert dv1 + dv2 <= sol16.delta_v_total_km_s
    assert sol16.delta_v_total_km_s <= 1.02 * edelbaum + 1e-6

    # 细化不离散变差（n=32 需更多 SLSQP 迭代：96 变量解析雅可比收敛耗时）。
    sol32 = problem.solve(32, guess="edelbaum", maxiter=800)
    assert sol32.status.name == "CONVERGED", sol32.message
    assert sol32.delta_v_total_km_s <= sol16.delta_v_total_km_s + 0.3


def test_analytic_jacobian_matches_finite_difference():
    """解析雅可比对中心差分（④ 机制：守 f & g STM → 约束雅可比推导链）。

    等式约束（6 维，位置/速度混合量纲已按问题尺度归一）逐元素对照
    atol 1e-5；段界不等式约束（n 维，原始量纲）atol 1e-4。三种参数化：
    常推力（质量链 + δ 项）；SEP 固定 ``r_helio_km``（非日心问题显式传
    常数，``dT/dr`` 恒为零——验证零链不引入伪导数）；SEP 日心语义
    （瞬时 |r|，真覆盖 ``dT/dr`` 功率衰减梯度链）。
    """
    departure = np.array([8000.0, 0.0, 0.0, 0.0, 6.0, 1.0])
    tof = 4000.0
    n = 4
    arrival = np.asarray(
        propagate_kepler_py(departure.tolist(), [tof], MU)["states"][0], dtype=float
    )
    x_test = np.tile(np.array([1.5e-3, -1.0e-3, 0.5e-3]), n)
    h = 1e-6

    # 日心 SEP 场景（1AU→1.5AU Hohmann 定相）：dT/dr 链的宿主工况。
    r1, r2 = AU_KM, 1.5 * AU_KM
    v1, v2 = float(np.sqrt(MU_SUN / r1)), float(np.sqrt(MU_SUN / r2))
    helio_problem = SimsFlanaganProblem(
        np.array([r1, 0.0, 0.0, 0.0, v1, 0.0]),
        np.array([-r2, 0.0, 0.0, 0.0, -v2, 0.0]),
        hohmann_tof(r1, r2, MU_SUN),
        SimsFlanaganPropulsion.sep(7.4e4, 3000.0),
        1000.0,
        MU_SUN,
        backend="conic",
    )

    cases = [
        (
            SimsFlanaganProblem(
                departure,
                arrival,
                tof,
                SimsFlanaganPropulsion.constant(5.0, 3000.0),
                1000.0,
                MU,
                backend="conic",
            ),
            x_test,
        ),
        (
            SimsFlanaganProblem(
                departure,
                arrival,
                tof,
                SimsFlanaganPropulsion.sep(1.0e5, 3000.0, r_helio_km=AU_KM),
                1000.0,
                MU,
                backend="conic",
            ),
            x_test,
        ),
        (helio_problem, x_test),
    ]
    for problem, x_case in cases:
        ev = problem._evaluate(x_case, n_segments=n, with_sens=True)
        assert ev.eq_jac is not None and ev.ineq_jac is not None
        for i in range(3 * n):
            x_plus = x_case.copy()
            x_plus[i] += h
            x_minus = x_case.copy()
            x_minus[i] -= h
            ev_plus = problem._evaluate(x_plus, n_segments=n, with_sens=False)
            ev_minus = problem._evaluate(x_minus, n_segments=n, with_sens=False)
            fd_eq = (ev_plus.eq - ev_minus.eq) / (2.0 * h)
            fd_ineq = (ev_plus.ineq - ev_minus.ineq) / (2.0 * h)
            np.testing.assert_allclose(fd_eq, ev.eq_jac[:, i], atol=1e-5)
            np.testing.assert_allclose(fd_ineq, ev.ineq_jac[:, i], atol=1e-4)
