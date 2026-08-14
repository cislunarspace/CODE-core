"""FiniteBurn 低推力传播物理验证。"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel
from e2m2e.algorithm.forces.force_config import build_force
from tests.numerical.forces.conftest import FakeSystem

pytestmark = pytest.mark.force


def test_finite_burn_vnb_spiral_raises_semi_major_axis(point_mass_force):
    """VNB 速度方向的恒质量推力使圆轨道半长轴持续提升。"""
    mu = point_mass_force.mu
    r0 = 6678.0
    v_circ = np.sqrt(mu / r0)
    y0 = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0])
    burn = build_force(
        "FiniteBurn",
        {
            "mass": 1000.0,
            "thrust_profile": {"kind": "constant", "thrust": 2.0},
            "direction": {"kind": "fixed", "vector": [1.0, 0.0, 0.0]},
            "direction_frame": "VNB",
        },
    )
    fm = ForceModel(FakeSystem(), forces=[point_mass_force, burn])

    period = 2.0 * np.pi * np.sqrt(r0**3 / mu)
    result = fm.propagate(y0, (0.0, 3.0 * period), t_eval=np.linspace(0.0, 3.0 * period, 60))

    states = result["states"]
    r = np.linalg.norm(states[:, :3], axis=1)
    v = np.linalg.norm(states[:, 3:], axis=1)
    energy = 0.5 * v**2 - mu / r
    semi_major = -mu / (2.0 * energy)

    assert semi_major[-1] > semi_major[0]
    assert np.polyfit(np.arange(semi_major.size), semi_major, 1)[0] > 0.0
