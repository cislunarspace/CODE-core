"""DPO 主族单圈分支回归（issue #841）。

旧种子（x0=0.90，T=2.5022）本身是双圈解，族行走从小振幅端就沿多圈
分支收敛：20000 km 返回 7 圈轨道（T≈23.3 天），21000/22000 km 返回
3/4 圈。主族单圈分支的自然参数延拓给出振幅折叠极值 27752 km，请求
包络取 27500 km。本文件钉住：20000 km 返回单圈主族成员、接近上界的
27500 km 仍收敛、主族成员整周期绕月恰好一圈。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import design_dpo
from e2m2e.algorithm.family.orbits.walk import _moon_distance_minmax, earth_moon_system
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


def _winding_number(orbit: Orbit) -> float:
    """整周期内绕月方位角的累计变化除以 2π（主族单圈顺行为 +1）。"""
    x_moon = 1.0 - orbit.system.mu
    angle = np.unwrap(np.arctan2(orbit.states[:, 1], orbit.states[:, 0] - x_moon))
    return float((angle[-1] - angle[0]) / (2.0 * np.pi))


def test_20000_km_returns_single_rev_principal_member() -> None:
    """20000 km 是 DPO 请求默认振幅，必须落在主族单圈分支（#841）。"""
    dynamics = CR3BP_Dynamics(earth_moon_system())
    lu = dynamics.system.characteristic_length
    assert lu is not None

    orbit = design_dpo(20_000.0, dynamics=dynamics)
    d_min, d_max = _moon_distance_minmax(dynamics, orbit)
    assert abs(0.5 * (d_min + d_max) * lu - 20_000.0) <= 20.0
    assert abs(_winding_number(orbit) - 1.0) < 0.05
    assert orbit.period is not None
    # 主族 20000 km 成员周期约 0.77 TU（3.33 天）；多圈伪解在 5 TU 以上，
    # 另一条单圈分支周期明显更长
    assert 0.70 < orbit.period < 0.85
    assert orbit.closure_error is not None
    assert orbit.closure_error <= 1.0e-10


def test_near_upper_bound_27500_km_converges() -> None:
    """请求包络上界 27500 km 距折叠极值 27752 km 约 250 km，仍应稳定收敛。"""
    dynamics = CR3BP_Dynamics(earth_moon_system())
    lu = dynamics.system.characteristic_length
    assert lu is not None

    orbit = design_dpo(27_500.0, dynamics=dynamics)
    d_min, d_max = _moon_distance_minmax(dynamics, orbit)
    assert abs(0.5 * (d_min + d_max) * lu - 27_500.0) <= 20.0
    assert abs(_winding_number(orbit) - 1.0) < 0.05
    # 近上界成员已相当偏心（近月点约 7000 km），仍在月面以上
    assert d_min * lu > 1737.0
    assert orbit.closure_error is not None
    assert orbit.closure_error <= 1.0e-10
