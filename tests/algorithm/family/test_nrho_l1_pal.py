"""L1 NRHO 单轨 PAL 路径回归（issue #772）。

L1 北族 Python PAL 延拓的方向反馈曾误翻转并锁存，叠加弧长约束未随
dir_sign 定向，步进陷入双能级 2-周期振荡，design_nrho(1, ...) 在 600
步内无法到达近月段。修复后两半球都经振幅增长支单次 PAL 越过折叠点。
此处锁端到端可达性：近月高命中容差、周期为正、一个周期传播闭合、
半球符号与请求一致；另验 PAL 提前终止的错误信息区分三种原因。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import design_nrho
from e2m2e.algorithm.family.orbits.nrho import _pal_termination_reason, _walk_pal_to_perilune
from e2m2e.algorithm.family.orbits.walk import Cr3bpOrbitError, earth_moon_system
from e2m2e.data.templates import MOON_RADIUS_KM

pytestmark = pytest.mark.orchestration


@pytest.mark.parametrize("north_south", [1, 2], ids=["north", "south"])
@pytest.mark.time_budget(120)
# 原因：L1 单次 PAL 需越过折叠点再二分细化，release 扩展实测约 4.4 s、本机 debug
# 扩展约 73 s；标记留 120 s 覆盖 debug 构建与慢机
def test_design_nrho_l1_hits_perilune(north_south: int) -> None:
    """L1 北/南族近月高 5000 km：命中 ±10 km、周期为正、闭合残差 ≤ 0.1 m。"""
    orbit = design_nrho(1, north_south, 5000.0)
    assert orbit.period is not None and orbit.period > 0.0

    system = orbit.system
    dynamics = CR3BP_Dynamics(system)
    lu = system.characteristic_length
    assert lu is not None
    result = dynamics.propagate(
        orbit.states[0], (0.0, orbit.period), t_eval=np.linspace(0.0, orbit.period, 1200)
    )
    states = np.asarray(result["states"])
    moon = np.array([1.0 - system.mu, 0.0, 0.0])
    r_moon_km = np.linalg.norm(states[:, :3] - moon, axis=1) * lu
    assert abs((r_moon_km.min() - MOON_RADIUS_KM) - 5000.0) <= 10.0
    closure_km = np.linalg.norm(states[-1, :3] - states[0, :3]) * lu
    assert closure_km <= 1.0e-4
    # 存储态半球符号与北/南请求一致
    assert orbit.states[0, 2] * (1.0 if north_south == 1 else -1.0) > 0.0


def test_walk_pal_family_end_message() -> None:
    """max_orbits 耗尽且成员单调推进时，错误报族已到尽头而非步进停滞。"""
    dynamics = CR3BP_Dynamics(system=earth_moon_system())
    lu = dynamics.system.characteristic_length
    assert lu is not None
    target_du = (5000.0 + MOON_RADIUS_KM) / lu
    with pytest.raises(Cr3bpOrbitError, match="族已到尽头"):
        _walk_pal_to_perilune(dynamics, 1, 1.0, target_du, 10.0 / lu, max_orbits=3)


def test_pal_termination_reason_classifies_sequences() -> None:
    """纯函数判定：单调序列报族尽头，往复振荡报步进停滞，空序列单独报。"""
    monotone = _pal_termination_reason([0.05 - 1e-3 * i for i in range(30)], 30, 0.017)
    assert "族已到尽头" in monotone and "停滞" not in monotone

    oscillating = _pal_termination_reason([0.0138, 0.0131] * 25, 50, 0.017)
    assert "步进停滞" in oscillating

    # 成员数少于请求条数：报提前中断，且条数按实际成员数给
    aborted = _pal_termination_reason([0.05 - 1e-3 * i for i in range(10)], 600, 0.017)
    assert "提前中断" in aborted and "10 条成员" in aborted and "600 条" in aborted

    empty = _pal_termination_reason([], 600, 0.017)
    assert "未产生" in empty
