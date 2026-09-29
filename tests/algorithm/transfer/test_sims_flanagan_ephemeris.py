"""Sims-Flanagan 星历 N 体档（backend="ephemeris"）验证（issue #727）。

oracle 口径（ADR 0055）：

- ③ 退化极限：单天体系（bodies=["EARTH"]，N 体公式退化为纯二体）下，固定
  控制向量的评估剖面与求解级结果同 conic 档逐项一致；
- ② 受力一致：EphemerisDynamics（Rust nbody_stm.rs）轨迹对 ForceModel 编译
  路径（compiled.rs / compute_total_acceleration 家族）的跨实现对拍
  （ADR 0034 决策 6(b) 先例）；
- ④ 机制接线：绝对历元对齐（前向/后向 pass 端点 vs 直接传播）、解析雅可比
  对中心差分（冲量列与 TOF 灵敏度链，TOF 链在时间不变的退化系验证——真实
  N 体下跨 leg 历元平移项属 #726 逐 leg 重定目标范围，不在本切片断言）。

⑦ 类文献结果性数值禁入 CI 断言。

μ 一致性（③ 的定义性前提）：conic 参考问题的 ``mu_km3_s2`` 一律取
``spice_manager.get_gm("EARTH")``（与星历档同一内核 GM），不用旧测试的
硬编码值——相对失配会在弧上放大为位置差，淹没待测的积分级差异。
"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import requires_native_symbols, requires_spice

from e2m2e.algorithm.dynamics import EphemerisDynamics, EphemerisSystem
from e2m2e.algorithm.transfer import (
    SimsFlanaganMultiLegProblem,
    SimsFlanaganNode,
    SimsFlanaganProblem,
    SimsFlanaganPropulsion,
)
from e2m2e.integrators import propagate_kepler_py

pytestmark = [
    pytest.mark.orchestration,
    pytest.mark.low_thrust,
    pytest.mark.spice,
    requires_spice,
    requires_native_symbols("propagate_kepler_py", "propagate_with_stm_py"),
]

#: LEC 场景参数（与 conic 档测试同型的 7000 km 圆轨道）。
_R0_KM = 7000.0
_TOF_S = 10800.0

#: 多 leg FD / parity 链参数（仿 test_sims_flanagan_multileg 的合成链拓扑）。
_ML_N_SEG = 4
_ML_DT = 1500.0
_V_IN = 2.0 * np.array([0.0, 0.0, 1.0])
_V_OUT = 2.0 * np.array([np.sin(np.radians(30.0)), 0.0, np.cos(np.radians(30.0))])
_R_P_MIN = 250000.0


@pytest.fixture
def eph_earth_only(spice_manager):
    """单天体退化星历系（N 体公式退化为纯二体：③ 退化极限宿主）。"""
    return EphemerisSystem(bodies=["EARTH"], spice=spice_manager, origin="EARTH")


def _leo_circular_departure(mu: float) -> np.ndarray:
    """7000 km 圆轨道出发态（赤道面内、x 轴起算）。"""
    return np.array([_R0_KM, 0.0, 0.0, 0.0, float(np.sqrt(mu / _R0_KM)), 0.0])


# =============================================================================
# ③ 退化极限：单天体系（纯二体）下与 conic 档逐项一致
# =============================================================================


def test_degenerate_fixed_vector_matches_conic(spice_manager, eph_earth_only, reference_et):
    """固定控制向量下，星历档评估剖面与 conic 档一致（③ 最锐对照）。

    同一决策向量 ``x_test``（逐段 0.01 km/s 冲量）喂给两档 ``_evaluate``：
    前向/后向节点状态与匹配点残差的差异只能来自积分截断（星历档半段
    span/10 步长上限下 ~1e-9 km，容差 1e-7 留两个量级裕度）。零冲量时
    弹道可达，残差为积分噪声级。
    """
    mu = float(spice_manager.get_gm("EARTH"))
    departure = _leo_circular_departure(mu)
    arrival = np.asarray(
        propagate_kepler_py(departure.tolist(), [_TOF_S], mu)["states"][0], dtype=float
    )
    prop = SimsFlanaganPropulsion.constant(5.0, 3000.0)
    conic = SimsFlanaganProblem(departure, arrival, _TOF_S, prop, 1000.0, mu, backend="conic")
    eph = SimsFlanaganProblem(
        departure,
        arrival,
        _TOF_S,
        prop,
        1000.0,
        mu,
        backend="ephemeris",
        epoch_et_s=reference_et,
        ephemeris_system=eph_earth_only,
    )
    n = 6
    x_test = np.tile(np.array([0.01, 0.0, 0.0]), n).reshape(-1)

    ev_conic = conic._evaluate(x_test, n_segments=n, with_sens=False)
    ev_eph = eph._evaluate(x_test, n_segments=n, with_sens=False)
    np.testing.assert_allclose(ev_eph.forward_states, ev_conic.forward_states, atol=1e-7)
    np.testing.assert_allclose(ev_eph.backward_states, ev_conic.backward_states, atol=1e-7)
    np.testing.assert_allclose(ev_eph.eq_raw, ev_conic.eq_raw, atol=1e-7)

    # 弹道可达：残差为星历积分噪声级（实测 2.1e-6 km；LEO 圆轨道 12 hop 的
    # 自适应接受步误差累积地板，比问题尺度低 3 个量级以上）。
    ev_ballistic = eph._evaluate(np.zeros(3 * n), n_segments=n, with_sens=False)
    assert float(np.max(np.abs(ev_ballistic.eq_raw))) < 1e-5


def test_degenerate_solve_matches_conic(spice_manager, eph_earth_only, reference_et):
    """求解级退化对照：星历档 NLP 收敛且 ΔV 与 conic 档一致（③）。"""
    mu = float(spice_manager.get_gm("EARTH"))
    departure = _leo_circular_departure(mu)
    arrival = np.asarray(
        propagate_kepler_py(departure.tolist(), [_TOF_S], mu)["states"][0], dtype=float
    )
    prop = SimsFlanaganPropulsion.constant(5.0, 3000.0)
    conic = SimsFlanaganProblem(departure, arrival, _TOF_S, prop, 1000.0, mu, backend="conic")
    eph = SimsFlanaganProblem(
        departure,
        arrival,
        _TOF_S,
        prop,
        1000.0,
        mu,
        backend="ephemeris",
        epoch_et_s=reference_et,
        ephemeris_system=eph_earth_only,
    )
    sol_conic = conic.solve(6)
    sol_eph = eph.solve(6)
    assert sol_eph.status.name == "CONVERGED", sol_eph.message
    assert sol_conic.status.name == "CONVERGED", sol_conic.message
    # 弹道可达问题：两档都收敛到零燃料附近，ΔV 差只含 SLSQP 路径噪声。
    assert abs(sol_eph.delta_v_total_km_s - sol_conic.delta_v_total_km_s) < 1e-4
    assert float(np.max(np.abs(sol_eph.matchpoint_residual))) < 1e-6
    assert float(np.max(np.abs(sol_conic.matchpoint_residual))) < 1e-6


# =============================================================================
# ② 受力一致：EphemerisDynamics vs ForceModel 编译路径（跨实现对拍）
# =============================================================================


def test_force_parity_with_force_model(spice_manager, spice_eph_system):
    """星历档内核与 ForceModel 编译路径的轨迹级受力对拍（②）。

    同一 3 体系（地心 origin + 月/日第三体）、同一初值与历元，DOP853
    （nbody_stm.rs，SF 星历档实际内核）vs PD45（compiled.rs）双实现传播；
    容差按两实现 1e-12 容差、2 h 弧的截断误差量级取值（A2 口径）。
    """
    from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
    from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
    from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
    from e2m2e.algorithm.forces import ForceModel, PointMassGravity, ThirdBodyGravity

    spice_eph_system.coordinate_system = CoordinateSystem(
        axes=ICRSAxes(),
        origin=CelestialBodyOrigin(body="EARTH", spice=spice_manager),
    )
    fm = ForceModel(
        spice_eph_system,
        [PointMassGravity("EARTH"), ThirdBodyGravity("MOON"), ThirdBodyGravity("SUN")],
    )
    dyn = EphemerisDynamics(system=spice_eph_system)

    state0 = np.array([8000.0, 1000.0, 0.0, 0.0, 6.9, 1.1])
    et0 = spice_manager.utc_to_et("2025-06-21T11:00:06")
    t_eval = [et0, et0 + 3600.0, et0 + 7200.0]

    out_fm = fm.propagate(state0, t_span=(et0, et0 + 7200.0), t_eval=t_eval)
    out_dyn = dyn.propagate(state0, (et0, et0 + 7200.0), t_eval)
    np.testing.assert_allclose(
        np.asarray(out_dyn["states"])[:, :3], np.asarray(out_fm["states"])[:, :3], atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(out_dyn["states"])[:, 3:], np.asarray(out_fm["states"])[:, 3:], atol=1e-7
    )


# =============================================================================
# ④ 机制接线：绝对历元对齐与解析雅可比
# =============================================================================


def test_absolute_epoch_wiring_ballistic(spice_manager, spice_eph_system, reference_et):
    """绝对历元接线：零冲量下 pass 端点与直接星历传播逐点对齐（④）。

    真实 3 体系（非退化）：前向 pass 覆盖前 ``m = n//2`` 段、后向 pass 覆盖
    后 ``n−m`` 段，两链在**同一绝对历元**（匹配点 ``et0 + m·dt``）会合——
    前向链末态对直接前向传播、后向链匹配点对直接后向传播（自 leg 末历元），
    双向共同验证绝对历元游标。参考传播收紧 max_step（10 s）使其截断 ≪
    pass 链；容差 1e-6 km 取 pass 链积分噪声（实测 ~2e-7 km）的余量。
    """
    mu = float(spice_manager.get_gm("EARTH"))
    departure = _leo_circular_departure(mu)
    tof = 3600.0
    et_end = reference_et + tof
    # 到达态须与 3 体动力学自洽：用紧参考前向传播生成（二体 Kepler 生成会
    # 引入 ~3e-3 km 的第三体摄动差，掩盖待测的历元接线精度）。
    ref = EphemerisDynamics(system=spice_eph_system)
    ref.max_step = 10.0
    arrival = np.asarray(ref.propagate(departure, (reference_et, et_end), [et_end])["states"][-1])
    problem = SimsFlanaganProblem(
        departure,
        arrival,
        tof,
        SimsFlanaganPropulsion.constant(5.0, 3000.0),
        1000.0,
        mu,
        backend="ephemeris",
        epoch_et_s=reference_et,
        ephemeris_system=spice_eph_system,
    )
    ev = problem._evaluate(np.zeros(12), n_segments=4, with_sens=False)
    m = 2  # n=4 → 前向 m=2 段、后向 2 段，匹配点在 tof/2 处
    et_match = reference_et + m * (tof / 4.0)

    fwd_ref = ref.propagate(departure, (reference_et, et_match), [et_match])["states"][-1]
    bwd_ref = ref.propagate(arrival, (et_end, et_match), [et_match])["states"][-1]

    np.testing.assert_allclose(ev.forward_states[-1], fwd_ref, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(ev.backward_states[-1], arrival, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(ev.backward_states[0], bwd_ref, rtol=0.0, atol=1e-6)
    # 两链在同一匹配点自洽（等式约束的原始残差，积分噪声级）。
    np.testing.assert_allclose(ev.forward_states[-1], ev.backward_states[0], rtol=0.0, atol=1e-5)


def test_ephemeris_analytic_jacobian_fd(spice_manager, spice_eph_system, reference_et):
    """星历档解析雅可比（数值 STM 链）对中心差分逐列对照（④ 机制）。"""
    mu = float(spice_manager.get_gm("EARTH"))
    departure = np.array([8000.0, 0.0, 0.0, 0.0, 6.0, 1.0])
    tof = 4000.0
    n = 4
    arrival = np.asarray(
        propagate_kepler_py(departure.tolist(), [tof], mu)["states"][0], dtype=float
    )
    problem = SimsFlanaganProblem(
        departure,
        arrival,
        tof,
        SimsFlanaganPropulsion.constant(5.0, 3000.0),
        1000.0,
        mu,
        backend="ephemeris",
        epoch_et_s=reference_et,
        ephemeris_system=spice_eph_system,
    )
    x_test = np.tile(np.array([1.5e-3, -1.0e-3, 0.5e-3]), n)
    # FD 步长 1e-4 km/s：星历档逐评估积分噪声 ~1e-6 km 且 FD 误差 ∝1/h
    # （h=1e-6 时噪声 ~2e-4 超 atol），h=1e-4 实测 |fd−解析| ~1.6e-6 ≪ 1e-5，
    # 非线性截断 O(h²) 仍可忽略。
    h = 1e-4
    ev = problem._evaluate(x_test, n_segments=n, with_sens=True)
    assert ev.eq_jac is not None and ev.ineq_jac is not None
    for i in range(3 * n):
        x_plus = x_test.copy()
        x_plus[i] += h
        x_minus = x_test.copy()
        x_minus[i] -= h
        ev_plus = problem._evaluate(x_plus, n_segments=n, with_sens=False)
        ev_minus = problem._evaluate(x_minus, n_segments=n, with_sens=False)
        fd_eq = (ev_plus.eq - ev_minus.eq) / (2.0 * h)
        fd_ineq = (ev_plus.ineq - ev_minus.ineq) / (2.0 * h)
        np.testing.assert_allclose(fd_eq, ev.eq_jac[:, i], atol=1e-5)
        np.testing.assert_allclose(fd_ineq, ev.ineq_jac[:, i], atol=1e-4)


def test_multileg_tof_jacobian_and_degenerate_parity(spice_manager, eph_earth_only, reference_et):
    """多 leg 星历档：TOF 链 FD + 退化 parity（③+④）。

    退化单天体 2-leg min_time 链（含 tof_bounds）验证冲量列与 TOF 列解析
    雅可比（时间不变退化系下跨 leg 历元平移无贡献，链结构可精确对照）；
    同一非零决策向量下 conic 与 ephemeris 的归一化等式约束一致。flyby 链
    变体加验 flyby 约束（等模/转角与传播无关，两档逐位一致）。
    """
    mu = float(spice_manager.get_gm("EARTH"))
    v0 = float(np.sqrt(mu / _R0_KM))
    dep = np.array([_R0_KM, 0.0, 0.0, 0.0, v0, 0.0])
    n, dt = _ML_N_SEG, _ML_DT
    tof_nom = n * dt
    w0 = np.tile(np.array([1.0e-3, -0.8e-3, 0.5e-3]), (n, 1))
    w1 = np.tile(np.array([0.6e-3, 0.3e-3, -0.4e-3]), (n, 1))

    def _kepler(x: np.ndarray, t: float) -> np.ndarray:
        return np.asarray(
            propagate_kepler_py(np.asarray(x, dtype=float).tolist(), [t], mu)["states"][0],
            dtype=float,
        )

    def _segmented(start: np.ndarray, impulses: np.ndarray) -> np.ndarray:
        x = start.copy()
        for dv_k in impulses:
            mid = _kepler(x, 0.5 * dt)
            mid[3:] += dv_k
            x = _kepler(mid, 0.5 * dt)
        return x

    a0 = _segmented(dep, w0)
    final = _segmented(a0.copy(), w1)

    common = dict(
        propulsion=SimsFlanaganPropulsion.constant(5.0, 3000.0),
        initial_mass_kg=1000.0,
        mu_km3_s2=mu,
    )
    nodes_rdv = [SimsFlanaganNode("rendezvous", a0), SimsFlanaganNode("rendezvous", final)]
    conic = SimsFlanaganMultiLegProblem(
        dep, nodes_rdv, [tof_nom, tof_nom], backend="conic", **common
    )
    eph = SimsFlanaganMultiLegProblem(
        dep,
        nodes_rdv,
        [tof_nom, tof_nom],
        backend="ephemeris",
        epoch_et_s=reference_et,
        ephemeris_system=eph_earth_only,
        **common,
    )

    # -- FD：冲量列（h=1e-6 km/s）与 TOF 列（h=1e-2 s）vs 解析雅可比 --
    layout = eph._make_layout([n, n], "min_time", None, tof_free=True)
    x_test = np.concatenate([w0.reshape(-1), w1.reshape(-1), [tof_nom * 0.97, tof_nom * 1.03]])
    ev = eph._evaluate(x_test, layout, with_sens=True)
    assert ev.eq_jac is not None and ev.ineq_jac is not None
    for i in range(layout.n_var):
        h = 1e-2 if i >= layout.tof_off else 1e-4  # 冲量列同上噪声口径
        x_plus = x_test.copy()
        x_plus[i] += h
        x_minus = x_test.copy()
        x_minus[i] -= h
        ev_plus = eph._evaluate(x_plus, layout, with_sens=False)
        ev_minus = eph._evaluate(x_minus, layout, with_sens=False)
        fd_eq = (ev_plus.eq - ev_minus.eq) / (2.0 * h)
        fd_ineq = (ev_plus.ineq - ev_minus.ineq) / (2.0 * h)
        np.testing.assert_allclose(fd_eq, ev.eq_jac[:, i], atol=1e-5)
        np.testing.assert_allclose(fd_ineq, ev.ineq_jac[:, i], atol=1e-4)

    # -- ③ parity：同决策向量下两档归一化等式约束一致 --
    ev_conic = conic._evaluate(x_test, layout, with_sens=False)
    np.testing.assert_allclose(ev.eq, ev_conic.eq, atol=1e-7)

    # -- flyby 链变体：中间 flyby 节点，flyby 约束与传播内核无关 --
    v_body = a0[3:] - _V_IN
    nodes_fb = [
        SimsFlanaganNode(
            "flyby", np.concatenate([a0[:3], v_body]), mu_km3_s2=mu, r_p_min_km=_R_P_MIN
        ),
        SimsFlanaganNode("rendezvous", final),
    ]
    conic_fb = SimsFlanaganMultiLegProblem(
        dep, nodes_fb, [tof_nom, tof_nom], backend="conic", **common
    )
    eph_fb = SimsFlanaganMultiLegProblem(
        dep,
        nodes_fb,
        [tof_nom, tof_nom],
        backend="ephemeris",
        epoch_et_s=reference_et,
        ephemeris_system=eph_earth_only,
        **common,
    )
    layout_fb = eph_fb._make_layout([n, n], "min_fuel", None, tof_free=False)
    x_fb = np.concatenate([w0.reshape(-1), w1.reshape(-1), _V_IN, _V_OUT])
    ev_fb = eph_fb._evaluate(x_fb, layout_fb, with_sens=False)
    ev_fb_conic = conic_fb._evaluate(x_fb, layout_fb, with_sens=False)
    np.testing.assert_allclose(ev_fb.eq, ev_fb_conic.eq, atol=1e-7)
    assert len(ev_fb.flybys) == len(ev_fb_conic.flybys) == 1
    assert ev_fb.flybys[0].delta == ev_fb_conic.flybys[0].delta
    assert ev_fb.flybys[0].eq_val == ev_fb_conic.flybys[0].eq_val
