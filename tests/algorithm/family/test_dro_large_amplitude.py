"""DRO 族生成大振幅端回归（issue #838 / ADR 0061）。

族生成窗口 475000 到 485000 km 对应周期约 27.29 天，是 110000 km 旧包络
之外、论文复现需要的区间。上限放宽到 480000 km 后，此用例锁住算法层
可达性：成员落入请求窗口、周期正确、闭合误差不超过 1e-10。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import design_dro_family
from e2m2e.algorithm.family.orbits.walk import _moon_distance_minmax, earth_moon_system
from e2m2e.status import ConvergenceState

pytestmark = pytest.mark.orchestration


# debug 构建实测约 43.1 s（n_orbits=2、单窗口族行走）。ADR 0037 的单测时间
# 门禁已于 2026-10-16 撤回，用例保留在默认套件以锁住 #838 的族生成可达性
# 与闭合误差，不是遍历性网格。
def test_dro_family_reaches_480000_km() -> None:
    dynamics = CR3BP_Dynamics(earth_moon_system())
    system = dynamics.system
    lu = system.characteristic_length
    tu = system.characteristic_time
    assert lu is not None
    assert tu is not None

    result = design_dro_family(475000.0, 485000.0, n_orbits=2, dynamics=dynamics)
    assert result.status is ConvergenceState.CONVERGED
    assert result.generated_members == result.requested_members == 2

    for orbit in result.family.orbits:
        d_min, d_max = _moon_distance_minmax(dynamics, orbit)
        amplitude_km = 0.5 * (d_min + d_max) * lu
        assert 475000.0 <= amplitude_km <= 485000.0
        assert orbit.period is not None
        period_days = orbit.period * tu / 86400.0
        assert 27.2 <= period_days <= 27.4
        assert orbit.periodicity_error is not None
        assert orbit.periodicity_error <= 1.0e-10
