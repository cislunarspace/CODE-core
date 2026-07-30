"""Q-law 低推力初猜生成器验证。

对照 ``docs/plans/qlaw-prd.md``：Q 单调下降、根数朝目标收敛、Q-law 初猜比
满推力初猜约束残差更小。纯二体（PointMassGravity，无需 SPICE）。
"""

import numpy as np

from e2m2e.core.forces import PointMassGravity
from e2m2e.transfer import EngineConfig, LowThrustShooting
from e2m2e.transfer.qlaw import qlaw_guess, rv_to_keplerian

MU = 398600.435507  # km³/s²，地球


def _system_forces():
    """纯二体地心系（SimpleNamespace，PointMassGravity，无需 SPICE）。"""
    from types import SimpleNamespace

    return SimpleNamespace(origin="EARTH"), [PointMassGravity("EARTH", mu=MU)]


def _semi_major_axis(state, mu):
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:6])
    return -mu / (2 * (v**2 / 2 - mu / r))


def test_rv_to_keplerian_circular_orbit():
    """rv_to_keplerian 对圆轨道返回 a 正确、e≈0。"""
    r0 = 7000.0
    v0 = np.sqrt(MU / r0)
    a, e, i, _raan, _w, _nu = rv_to_keplerian(
        np.array([r0, 0.0, 0.0]), np.array([0.0, v0, 0.0]), MU
    )
    assert abs(a - r0) / r0 < 1e-10
    assert e < 1e-6
    assert abs(i) < 1e-10


def test_qlaw_q_monotone_decrease():
    """Q-law 前向积分 Q 单调下降（圆-圆共面抬高）。

    Q-law 是 Lyapunov 反馈律，Q(t) 应非增（容许数值噪声小波动）。
    """
    system, forces = _system_forces()
    r0 = 7000.0
    v0 = np.sqrt(MU / r0)
    init = np.array([r0, 0, 0, 0, v0, 0])
    engine = EngineConfig(t_max=0.5, isp=3000.0)
    # 目标 8000km 圆轨道（5 天足够显著移动）
    _y, _segs, qh, _final = qlaw_guess(
        system,
        forces,
        engine,
        init,
        1000.0,
        (8000.0, 0.0, 0.0),
        0.0,
        5 * 86400.0,
        5,
        step=180.0,
    )
    # Q 整体下降（首 > 末）
    assert qh[-1] < qh[0], f"Q 应单调下降: 首 {qh[0]:.3e} 末 {qh[-1]:.3e}"
    # 大部分相邻段非增（容许个别小波动）
    diffs = np.diff(qh)
    assert np.sum(diffs <= 1e-6) >= 0.7 * len(diffs), (
        f"Q 应大体非增, 正增量占比过高: {np.sum(diffs > 1e-6)}/{len(diffs)}"
    )


def test_qlaw_semi_major_axis_converges():
    """Q-law 前向积分末态 a 朝目标显著移动（final_state 是连续反馈真实结果）。"""
    system, forces = _system_forces()
    r0 = 7000.0
    v0 = np.sqrt(MU / r0)
    init = np.array([r0, 0, 0, 0, v0, 0])
    engine = EngineConfig(t_max=0.5, isp=3000.0)
    _y, _segs, _qh, final = qlaw_guess(
        system,
        forces,
        engine,
        init,
        1000.0,
        (8000.0, 0.0, 0.0),
        0.0,
        5 * 86400.0,
        5,
        step=180.0,
    )
    a_final = _semi_major_axis(final, MU)
    # 5 天应至少朝 8000 移动 200km（从 7000 起）
    assert a_final > r0 + 200, f"Q-law 应提升 a: a_final={a_final:.1f} (初 {r0}, 目标 8000)"
    # 偏心率不应被显著引入（圆-圆）
    a_k, e_k, *_ = rv_to_keplerian(final[:3], final[3:6], MU)
    assert e_k < 0.05, f"圆-圆转移不应引入大偏心率: e={e_k:.4f}"


def test_qlaw_better_initial_guess_than_full_thrust():
    """Q-law 初猜比满推力初猜更接近可行（约束残差更小）。

    满推力沿初速方向「推过头」，约束残差大；Q-law 跟随轨道，残差小。
    """
    system, forces = _system_forces()
    r0 = 7000.0
    v0 = np.sqrt(MU / r0)
    aT = 7500.0
    vT = np.sqrt(MU / aT)
    init = np.array([r0, 0, 0, 0, v0, 0])
    target = np.array([aT, 0, 0, 0, vT, 0])
    engine = EngineConfig(t_max=0.5, isp=3000.0)
    tf = 3 * 86400.0
    n_seg = 5

    # Q-law 初猜
    y_qlaw, _segs, _qh, _final = qlaw_guess(
        system,
        forces,
        engine,
        init,
        1000.0,
        (aT, 0.0, 0.0),
        0.0,
        tf,
        n_seg,
        step=180.0,
    )
    # 满推力初猜（throttle=1 沿初速方向 → θ₁=π/2, θ₂=0）
    y_full = np.tile(np.array([1.0, np.pi / 2, 0.0]), n_seg)

    # 用求解器重建约束残差（段内固定方向，两者同等条件）
    shooter = LowThrustShooting(system, forces, engine, init, 1000.0, target, 0.0, tf)
    res_qlaw = np.linalg.norm(shooter._terminal_constraint(y_qlaw))
    res_full = np.linalg.norm(shooter._terminal_constraint(y_full))
    assert res_qlaw < res_full, f"Q-law 初猜约束残差应更小: qlaw={res_qlaw:.4f} full={res_full:.4f}"
