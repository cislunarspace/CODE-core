"""恒星共振轨道（RO）设计入口的功能测试（issue #627）。

RO = 绕地质心的顺行平面周期轨道，p:q = 航天器惯性圈数:月球圈数。
会合系闭合周期为 ``T = 2πq/(p−q)``；设计路径按共振比选择近圆或偏心
种子，修正出精确成员，指定振幅时再以 +x 轴穿越点 x0 沿族行走。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import registry
from e2m2e.algorithm.family.cr3bp_orbits import (
    _earth_distance_minmax,
    design_ro,
    design_ro_family,
    earth_moon_system,
)
from e2m2e.data.templates import RO_SUPPORTED_RESONANCES, ConvergenceState

pytestmark = pytest.mark.orchestration

#: 3:1 精确共振成员的标称振幅（km，距地心距离 min/max 均值，实测值）
_RO_31_ANCHOR_AMPLITUDE_KM = 183810.0


def _amplitude_km(dynamics: CR3BP_Dynamics, orbit) -> float:
    """距地心距离 min/max 均值（km），与设计入口同一测量函数。"""
    d_min, d_max = _earth_distance_minmax(dynamics, orbit)
    return 0.5 * (d_min + d_max) * dynamics.system.characteristic_length


@pytest.mark.parametrize(("p", "q"), sorted(RO_SUPPORTED_RESONANCES))
def test_design_ro_exact_resonance_member(p: int, q: int) -> None:
    """各支持共振比的精确成员：周期、闭合与 Kepler 半长轴量级达标。"""
    dynamics = CR3BP_Dynamics(earth_moon_system())
    orbit = design_ro(p, q, dynamics=dynamics)
    expected_period = 2.0 * np.pi * q / (p - q)
    assert orbit.period is not None
    assert abs(orbit.period - expected_period) / expected_period < 1e-9, (
        f"{p}:{q} 周期 {orbit.period:.6f} 偏离精确通约值 {expected_period:.6f}"
    )
    assert orbit.closure_error is not None and orbit.closure_error < 1e-6
    assert orbit.is_periodic
    measured_du = _amplitude_km(dynamics, orbit) / dynamics.system.characteristic_length
    a_kepler = ((1.0 - dynamics.system.mu) * (q / p) ** 2) ** (1.0 / 3.0)
    assert abs(measured_du - a_kepler) / a_kepler <= 0.1


def test_design_ro_rejects_unsupported_resonance() -> None:
    """不支持（或非法）的共振比明确拒绝，不静默改设计别的轨道。"""
    with pytest.raises(ValueError, match="不支持的共振比 1:2"):
        design_ro(1, 2)


def test_design_ro_amplitude_target_walks_family() -> None:
    """指定振幅时沿族行走命中目标：振幅在容差内、轨道仍严格闭合。"""
    dynamics = CR3BP_Dynamics(earth_moon_system())
    target_km = _RO_31_ANCHOR_AMPLITUDE_KM + 700.0
    orbit = design_ro(3, 1, amplitude_km=target_km, tol_km=100.0, dynamics=dynamics)
    assert orbit.closure_error is not None and orbit.closure_error < 1e-6
    measured = _amplitude_km(dynamics, orbit)
    assert abs(measured - target_km) <= 100.0, (
        f"振幅 {measured:.0f} km 未命中目标 {target_km:.0f} km"
    )
    # 行走离开精确通约点，周期随振幅漂移。
    assert orbit.period is not None
    assert abs(orbit.period - np.pi) > 0.01


def test_design_ro_default_returns_exact_member() -> None:
    """不指定振幅时返回 3:1 精确成员。"""
    dynamics = CR3BP_Dynamics(earth_moon_system())
    orbit = design_ro(3, 1, dynamics=dynamics)
    assert abs(orbit.period - np.pi) < 1e-9
    measured = _amplitude_km(dynamics, orbit)
    assert abs(measured - _RO_31_ANCHOR_AMPLITUDE_KM) < 100.0


def test_registry_dispatches_ro() -> None:
    """族注册表含 RO 条目，按请求参数形状分发。"""
    assert "RO" in registry
    orbit = registry["RO"](resonance_p=3, resonance_q=1)
    assert orbit.period is not None
    assert abs(orbit.period - np.pi) < 1e-9


def test_design_ro_family_continuation_chain() -> None:
    """RO 族延拓链：窗口内多成员、按振幅升序、相邻成员自然延拓连续。"""
    result = design_ro_family(3, 1, 180000.0, 190000.0, n_orbits=4)
    assert result.status is ConvergenceState.CONVERGED, result.message
    family = result.family
    assert family.family_type == "ro"
    members = list(family.orbits)
    assert len(members) >= 3
    amplitudes = [orbit.parameters["amplitude_km"] for orbit in members]
    assert amplitudes == sorted(amplitudes)
    assert all(180000.0 <= amp <= 190000.0 for amp in amplitudes)
    # 相邻成员自然延拓：x0 步长即名义延拓步长（0.005），链上无跳支
    x0s = [float(orbit.states[0, 0]) for orbit in members]
    assert x0s == sorted(x0s)  # 振幅随 x0 单调
    gaps = np.diff(x0s)
    assert np.all(gaps <= 0.005 + 1e-12)
    # 全部成员严格周期闭合
    for orbit in members:
        assert orbit.closure_error is not None and orbit.closure_error < 1e-6
        assert orbit.period is not None
    # 精确共振成员在链上（窗口跨种子振幅）
    assert any(abs(orbit.period - np.pi) < 1e-6 for orbit in members), (
        "族链应包含 3:1 精确共振成员（T = T_moon/2）"
    )
    # 成员参数携带共振比 provenance
    assert all(orbit.parameters["resonance_p"] == 3 for orbit in members)


def test_design_ro_family_rejects_unsupported_resonance() -> None:
    with pytest.raises(ValueError, match="不支持的共振比"):
        design_ro_family(1, 2, 180000.0, 190000.0)
