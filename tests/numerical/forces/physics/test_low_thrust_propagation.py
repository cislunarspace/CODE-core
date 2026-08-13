"""低推力传播端到端验证。

验证半长轴变化率与解析公式误差 < 5%、7 天螺旋轨道演化。
FiniteBurn 配置 round-trip 见 ``config/test_low_thrust_config_propagation.py``。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import FiniteBurn, ForceModel, GravityField
from tests.numerical.forces.conftest import (
    EARTH_RE,
    keplerian_to_cartesian,
    semi_major_axis,
)

pytestmark = [pytest.mark.force, pytest.mark.low_thrust]


@pytest.mark.spice
@pytest.mark.xfail(reason="预留 #407：FiniteBurn 恒质量低推力从未实现")
def test_low_thrust_circular_orbit_semi_major_axis_rate(earth_icrf_system):
    """低推力圆轨道提升：半长轴变化率与解析公式误差 < 5%。"""
    system = earth_icrf_system
    mu = system.gravitational_parameter("EARTH")

    r_earth = EARTH_RE
    a0 = r_earth + 300.0
    y0 = keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    thrust = 0.1  # N
    mass = 1000.0  # kg
    duration_s = 1.0 * 86400.0

    def thrust_profile(_t: float) -> float:
        return thrust

    def direction(_t: float, state: np.ndarray) -> np.ndarray:
        v = state[3:6]
        return v / np.linalg.norm(v)

    burn = FiniteBurn(thrust_profile=thrust_profile, direction=direction, mass=mass)
    gravity = GravityField(body="EARTH", degree=2, order=0)
    fm = ForceModel(system, forces=[gravity, burn])

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.linspace(et0, et0 + duration_s, 100)

    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    a_history = np.array([semi_major_axis(s, mu) for s in result["states"]])
    a_final = a_history[-1]

    # 解析解：a(t) = (a0^(3/2) + 3/2 * sqrt(mu) * (F/m) * t)^(2/3)
    a_theory = (a0**1.5 + 1.5 * np.sqrt(mu) * (thrust / mass) * duration_s) ** (2.0 / 3.0)

    relative_error = abs((a_final - a_theory) / a_theory)
    assert relative_error < 0.05, (
        f"半长轴变化率偏差过大: measured={a_final:.3f} km, "
        f"theory={a_theory:.3f} km, error={relative_error:.1%}"
    )
