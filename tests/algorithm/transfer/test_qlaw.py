"""Q-law 低推力初猜生成器验证。

对照 ``docs/plans/qlaw-prd.md``：Q 值下降与轨道根数朝目标收敛。纯二体
（PointMassGravity，无需 SPICE）。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import PointMassGravity
from e2m2e.algorithm.transfer import EngineConfig
from e2m2e.algorithm.transfer.qlaw import qlaw_guess, rv_to_keplerian

pytestmark = [pytest.mark.orchestration, pytest.mark.low_thrust]


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


@pytest.fixture(scope="module")
def qlaw_short_arc():
    """一次短弧 Q-law 传播，为两个物理不变量复用结果。"""
    system, forces = _system_forces()
    r0 = 7000.0
    v0 = np.sqrt(MU / r0)
    initial = np.array([r0, 0, 0, 0, v0, 0])
    engine = EngineConfig(t_max=0.5, isp=3000.0)
    _y, _segments, q_history, final_state = qlaw_guess(
        system,
        forces,
        engine,
        initial,
        1000.0,
        (7020.0, 0.0, 0.0),
        0.0,
        6 * 3600.0,
        3,
        step=600.0,
    )
    return q_history, final_state


def test_qlaw_q_monotone_decrease(qlaw_short_arc):
    """Q-law 短弧反馈传播的 Q 值大体下降。"""
    q_history, _ = qlaw_short_arc

    assert q_history[-1] < q_history[0], f"Q 应下降: 首 {q_history[0]:.3e} 末 {q_history[-1]:.3e}"
    diffs = np.diff(q_history)
    assert np.sum(diffs <= 1e-6) >= 0.7 * len(diffs), (
        f"Q 应大体非增, 正增量占比过高: {np.sum(diffs > 1e-6)}/{len(diffs)}"
    )


def test_qlaw_semi_major_axis_converges(qlaw_short_arc):
    """Q-law 短弧反馈传播推动半长轴向目标移动。"""
    _, final = qlaw_short_arc
    r0 = 7000.0

    a_final = _semi_major_axis(final, MU)
    assert a_final > r0 + 15, f"Q-law 应提升 a: a_final={a_final:.1f} (初 {r0})"
    _a, eccentricity, *_ = rv_to_keplerian(final[:3], final[3:6], MU)
    assert eccentricity < 0.05, f"圆-圆转移不应引入大偏心率: e={eccentricity:.4f}"
