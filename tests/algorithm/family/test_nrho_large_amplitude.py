"""9:2 量级大振幅北族 NRHO 单轨设计回归（issue #643）。

近月高 14 870 km 的 L2 北族成员此前被设计侧 10 000 km 上限拒绝（该量级
对应 9:2 共振 NRHO，z0≈77 787 km 恰在 Halo 族 z0 折叠顶附近，只能经
NRHO 近月高路径逼近）。上限放宽到 40 000 km 后，此用例锁住算法层可达
性：近月高命中容差、周期为正、一个周期传播闭合。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import design_nrho
from e2m2e.algorithm.family.cr3bp_orbits import MOON_RADIUS_KM

pytestmark = pytest.mark.orchestration


@pytest.mark.time_budget(70)
def test_design_nrho_9_2_magnitude_north() -> None:
    """L2 北族近月高 14 870 km：命中 ±10 km、周期为正、闭合残差 ≤ 0.1 m。"""
    orbit = design_nrho(2, 1, 14870.0)
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
    assert abs((r_moon_km.min() - MOON_RADIUS_KM) - 14870.0) <= 10.0

    closure_km = np.linalg.norm(states[-1, :3] - states[0, :3]) * lu
    assert closure_km <= 1.0e-4
