"""FiniteBurn 配置往返的物理验收（ADR 0004）。

``to_config`` → ``from_config`` 重建的力模型与原始力模型传播同一段低推力
轨道，末态一致（< 1e-12）。这是配置保真的物理检查，与
``test_leo_config_propagation.py``（LEO 场景）同类。

低推力功能尚未开发完成，本文件标记 ``low_thrust``，本轮检查排除在绿门外。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import FiniteBurn, ForceModel, GravityField
from tests.numerical.forces.conftest import EARTH_RE, keplerian_to_cartesian

pytestmark = [pytest.mark.force, pytest.mark.low_thrust]


@pytest.mark.spice
@pytest.mark.xfail(reason="预留 #407：FiniteBurn 恒质量低推力从未实现")
def test_low_thrust_config_round_trip(earth_icrf_system):
    """FiniteBurn 配置往返后，低推力传播结果一致。"""
    system = earth_icrf_system
    mu = system.gravitational_parameter("EARTH")

    r_earth = EARTH_RE
    a0 = r_earth + 300.0
    y0 = keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

    thrust = 0.1
    mass = 1000.0
    duration_s = 1.0 * 86400.0

    from e2m2e.algorithm.forces.force_config import build_force

    burn_original = FiniteBurn(
        thrust_profile=build_force(
            "FiniteBurn",
            {
                "thrust_profile": {"kind": "constant", "thrust": thrust},
                "direction": {"kind": "fixed", "vector": [0.0, 1.0, 0.0]},
                "mass": mass,
            },
        ).thrust_profile,
        direction=[0.0, 1.0, 0.0],
        mass=mass,
    )
    fm_original = ForceModel(
        system,
        forces=[GravityField(body="EARTH", degree=0, order=0), burn_original],
    )

    config = fm_original.to_config()
    fm_restored = ForceModel.from_config(config, system)

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + duration_s)
    t_eval = np.array([et0, et0 + duration_s])

    result_original = fm_original.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)
    result_restored = fm_restored.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    np.testing.assert_allclose(
        result_original["states"][-1],
        result_restored["states"][-1],
        rtol=1e-12,
        atol=1e-12,
    )
