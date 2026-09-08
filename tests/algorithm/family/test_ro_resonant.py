"""共振轨道（RO）设计入口的功能测试（issue #627）。

RO = 绕地质心的顺行近圆周期轨道，p:q = 卫星:月球（旋转系周期
T = (q/p)·T☾，与分类学 resonant_p_q 及 ADR 0042 一致）。设计路径：
共振周期条件构造 Kepler 圆轨道初猜 → 固定半周期的 x 轴对称修正出
精确通约成员 → 指定振幅时以 +x 轴穿越点 x0 为族参数自然延拓命中目标。
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
_RO_31_ANCHOR_AMPLITUDE_KM = 151818.0


def _amplitude_km(dynamics: CR3BP_Dynamics, orbit) -> float:
    """距地心距离 min/max 均值（km），与设计入口同一测量函数。"""
    d_min, d_max = _earth_distance_minmax(dynamics, orbit)
    return 0.5 * (d_min + d_max) * dynamics.system.characteristic_length


@pytest.mark.parametrize(("p", "q"), sorted(RO_SUPPORTED_RESONANCES))
def test_design_ro_exact_resonance_member(p: int, q: int) -> None:
    """各支持共振比的精确通约成员：周期恰为 (q/p)·T☾，闭合残差 < 1e-6。"""
    orbit = design_ro(p, q)
    expected_period = (q / p) * 2.0 * np.pi
    assert orbit.period is not None
    assert abs(orbit.period - expected_period) / expected_period < 1e-9, (
        f"{p}:{q} 周期 {orbit.period:.6f} 偏离精确通约值 {expected_period:.6f}"
    )
    assert orbit.closure_error is not None and orbit.closure_error < 1e-6
    assert orbit.is_periodic


def test_design_ro_rejects_unsupported_resonance() -> None:
    """不支持（或非法）的共振比明确拒绝，不静默改设计别的轨道。"""
    with pytest.raises(ValueError, match="不支持的共振比 1:2"):
        design_ro(1, 2)


def test_design_ro_amplitude_target_walks_family() -> None:
    """指定振幅时沿族行走命中目标：振幅在容差内、轨道仍严格闭合。

    目标取种子振幅近邻（152,500 km ≈ 锚点 +700 km）、筛选级容差
    （tol_km=100）：族行走按"命中容差即停"语义在少数几步内收敛，
    单测预算内（ADR 0037 决策 3）。
    """
    dynamics = CR3BP_Dynamics(earth_moon_system())
    target_km = 152500.0
    orbit = design_ro(3, 1, amplitude_km=target_km, tol_km=100.0, dynamics=dynamics)
    assert orbit.closure_error is not None and orbit.closure_error < 1e-6
    measured = _amplitude_km(dynamics, orbit)
    assert abs(measured - target_km) <= 100.0, (
        f"振幅 {measured:.0f} km 未命中目标 {target_km:.0f} km"
    )
    # 行走离开精确通约点，周期随振幅漂移（不再等于 T☾/3）
    assert orbit.period is not None
    assert abs(orbit.period - 2.0 * np.pi / 3.0) > 0.01


def test_design_ro_default_returns_exact_member() -> None:
    """不指定振幅时返回精确共振成员（3:1 锚点振幅即精确成员的振幅）。"""
    dynamics = CR3BP_Dynamics(earth_moon_system())
    orbit = design_ro(3, 1, dynamics=dynamics)
    assert abs(orbit.period - 2.0 * np.pi / 3.0) < 1e-9
    measured = _amplitude_km(dynamics, orbit)
    assert abs(measured - _RO_31_ANCHOR_AMPLITUDE_KM) < 100.0


def test_registry_dispatches_ro() -> None:
    """族注册表含 RO 条目，按请求参数形状（resonance_p/q + amplitude）分发。"""
    assert "RO" in registry
    orbit = registry["RO"](resonance_p=3, resonance_q=1)
    assert orbit.period is not None
    assert abs(orbit.period - 2.0 * np.pi / 3.0) < 1e-9


def test_design_ro_family_continuation_chain() -> None:
    """RO 族延拓链：窗口内多成员、按振幅升序、相邻成员自然延拓连续。"""
    result = design_ro_family(3, 1, 148000.0, 156000.0, n_orbits=4)
    assert result.status is ConvergenceState.CONVERGED, result.message
    family = result.family
    assert family.family_type == "ro"
    members = list(family.orbits)
    assert len(members) >= 3
    amplitudes = [orbit.parameters["amplitude_km"] for orbit in members]
    assert amplitudes == sorted(amplitudes)
    assert all(148000.0 <= amp <= 156000.0 for amp in amplitudes)
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
    assert any(abs(orbit.period - 2.0 * np.pi / 3.0) < 1e-6 for orbit in members), (
        "族链应包含 3:1 精确共振成员（T = T☾/3）"
    )
    # 成员参数携带共振比 provenance
    assert all(orbit.parameters["resonance_p"] == 3 for orbit in members)


def test_design_ro_family_rejects_unsupported_resonance() -> None:
    with pytest.raises(ValueError, match="不支持的共振比"):
        design_ro_family(1, 2, 148000.0, 156000.0)
