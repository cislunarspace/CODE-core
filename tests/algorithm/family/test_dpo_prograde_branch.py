"""DPO 顺行分支校验（issue #587）。

DPO 定义为绕月顺行轨道：近侧 x 轴穿越点 vy0 < 0，月心角动量
h_z = (x0 − x_moon)·vy0 > 0。固定 x0 的对称修正对同一 x0 存在顺行与
逆行两支解，收敛到哪支取决于初猜；packaged baseline dpo 曾因此混入
逆行成员（ADR 0042 Reproduction notes）。本文件钉住：DPO 修正拒绝
逆行支收敛解，``design_dpo`` 返回的成员全部顺行。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family.cr3bp_orbits import (
    Cr3bpOrbitError,
    _correct_dpo,
    design_dpo,
    earth_moon_system,
)
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


def _is_prograde(orbit: Orbit) -> bool:
    """穿越点月心角动量顺行（h_z > 0，等价近侧穿越点 vy0 < 0）。"""
    x_moon = 1.0 - orbit.system.mu
    return (float(orbit.states[0, 0]) - x_moon) * float(orbit.states[0, 4]) > 0.0


def _retrograde_guess(dynamics: CR3BP_Dynamics) -> Orbit:
    """近月逆行支上的初猜：小逆行月心轨道（打包基线 m0 附近真解的邻域）。"""
    state = np.array([0.9789, 0.0, 0.0, 0.0, 1.17, 0.0])
    orbit = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    orbit.period = 0.048
    return orbit


def test_correct_dpo_rejects_retrograde_branch() -> None:
    """逆行支初猜的修正应按伪解同路径拒绝（族行走退半步重试），而非返回逆行轨道。"""
    dynamics = CR3BP_Dynamics(earth_moon_system())
    with pytest.raises(Cr3bpOrbitError, match="逆行"):
        _correct_dpo(dynamics, 0.9789, _retrograde_guess(dynamics))


@pytest.mark.parametrize("amplitude_km", [2000.0, 5000.0, 15000.0])
def test_design_dpo_returns_prograde_orbit(amplitude_km: float) -> None:
    """支持振幅域内的目标，``design_dpo`` 返回的轨道必须顺行（#587 门面契约）。

    旧实现族行走会跳到逆行支（打包 baseline 前 4 成员为逆行，小振幅端
    全部中招）；顺行分支校验后，行走遇逆行解退半步重试，全域不再回退。
    """
    orbit = design_dpo(amplitude_km)
    assert orbit.period is not None
    assert _is_prograde(orbit), (
        f"design_dpo({amplitude_km:.0f} km) 返回逆行轨道（vy0={float(orbit.states[0, 4]):+.4f}）"
    )
