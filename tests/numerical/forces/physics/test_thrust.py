"""FiniteBurn 低推力传播物理验证。

沿速度方向低推力传播 → 半长轴持续提升（电推进螺旋趋势）。定义契约见
``contract/test_thrust.py``。

低推力功能尚未开发完成（Facade 任务入口占位，Rust 传播路径已接入）；本文件标记
``low_thrust``，本轮检查排除在绿门外。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel
from e2m2e.algorithm.forces.thrust import FiniteBurn
from tests.numerical.forces.conftest import FakeSystem

pytestmark = [pytest.mark.force, pytest.mark.low_thrust]


@pytest.mark.xfail(reason="预留 #407：FiniteBurn 恒质量低推力从未实现")
def test_finite_burn_spiral_raises_semi_major_axis(point_mass_force):
    """沿速度方向低推力传播 → 半长轴持续提升（电推进螺旋趋势）。"""
    mu = point_mass_force.mu
    r0 = 6678.0  # LEO 300km
    v_circ = np.sqrt(mu / r0)
    y0 = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0])

    burn = FiniteBurn(
        thrust_profile=lambda t: 2.0,  # N
        direction=lambda t, s: s[3:6],  # 沿速度方向（可调用）
        mass=1000.0,  # kg
    )
    fm = ForceModel(FakeSystem(), forces=[point_mass_force, burn])

    period = 2.0 * np.pi * np.sqrt(r0**3 / mu)
    n_orbits = 3
    result = fm.propagate(
        y0, (0.0, n_orbits * period), t_eval=np.linspace(0.0, n_orbits * period, 60)
    )

    states = result["states"]
    r = np.linalg.norm(states[:, :3], axis=1)
    v = np.linalg.norm(states[:, 3:], axis=1)
    energy = 0.5 * v**2 - mu / r
    semi_major = -mu / (2.0 * energy)

    # 半长轴末端 > 起点
    assert semi_major[-1] > semi_major[0]
    # 整体趋势向上（线性拟合斜率 > 0）
    slope = np.polyfit(np.arange(semi_major.size), semi_major, 1)[0]
    assert slope > 0.0
