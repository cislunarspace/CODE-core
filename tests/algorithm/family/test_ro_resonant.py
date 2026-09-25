"""恒星共振轨道（RO）设计入口的功能测试（issue #627）。

RO = 绕地质心的顺行平面周期轨道，p:q = 航天器惯性圈数:月球圈数：
q 个恒星月内绕地 p 圈（Vaquero & Howell 2014 式（7）），会合系周期
T = 2πq、净卷绕 w = p−q，与 resonant.csv 目录各族的精确成员同支。
设计路径按共振比选偏心族入口，割线钉定精确周期；指定振幅时再以
+x 轴穿越点 x0 沿族行走。
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
_RO_31_ANCHOR_AMPLITUDE_KM = 180113.0
#: 3:2 精确共振成员的标称振幅（同口径实测值）
_RO_32_ANCHOR_AMPLITUDE_KM = 294506.0


def _amplitude_km(dynamics: CR3BP_Dynamics, orbit) -> float:
    """距地心距离 min/max 均值（km），与设计入口同一测量函数。"""
    d_min, d_max = _earth_distance_minmax(dynamics, orbit)
    return 0.5 * (d_min + d_max) * dynamics.system.characteristic_length


def _winding_revolutions(orbit) -> float:
    """会合系净卷绕数（绕质心，单位 2π），从修正轨迹采样直接测量。"""
    theta = np.unwrap(np.arctan2(orbit.states[:, 1], orbit.states[:, 0]))
    return float((theta[-1] - theta[0]) / (2.0 * np.pi))


@pytest.mark.parametrize(("p", "q"), sorted(RO_SUPPORTED_RESONANCES))
def test_design_ro_exact_resonance_member(p: int, q: int) -> None:
    """精确成员：周期 T=2πq、卷绕 w=p−q、闭合与 Kepler 半长轴量级达标。"""
    dynamics = CR3BP_Dynamics(earth_moon_system())
    orbit = design_ro(p, q, dynamics=dynamics)
    expected_period = 2.0 * np.pi * q
    assert orbit.period is not None
    assert abs(orbit.period - expected_period) / expected_period < 1e-9, (
        f"{p}:{q} 周期 {orbit.period:.6f} 偏离目标 {expected_period:.6f}"
    )
    assert orbit.closure_error is not None and orbit.closure_error < 1e-6
    assert orbit.is_periodic
    assert abs(_winding_revolutions(orbit) - (p - q)) < 0.01
    measured_du = _amplitude_km(dynamics, orbit) / dynamics.system.characteristic_length
    a_kepler = ((1.0 - dynamics.system.mu) * (q / p) ** 2) ** (1.0 / 3.0)
    assert abs(measured_du - a_kepler) / a_kepler <= 0.1


def test_design_ro_rejects_unsupported_resonance() -> None:
    """不支持（或非法）的共振比明确拒绝，不静默改设计别的轨道。"""
    with pytest.raises(ValueError, match="不支持的共振比 1:2"):
        design_ro(1, 2)


def test_design_ro_amplitude_target_walks_family() -> None:
    """指定振幅时沿族行走命中目标：振幅在容差内、轨道仍严格闭合。

    选 3:2 族（x0–vy0 映射平缓，行走区实测稳定）；3:1/4:1 的陡峭区
    行走由割线逐点重建种子承担，此处只需证明行走机制本身。
    """
    dynamics = CR3BP_Dynamics(earth_moon_system())
    target_km = _RO_32_ANCHOR_AMPLITUDE_KM - 700.0
    orbit = design_ro(3, 2, amplitude_km=target_km, tol_km=100.0, dynamics=dynamics)
    assert orbit.closure_error is not None and orbit.closure_error < 1e-6
    measured = _amplitude_km(dynamics, orbit)
    assert abs(measured - target_km) <= 100.0, (
        f"振幅 {measured:.0f} km 未命中目标 {target_km:.0f} km"
    )
    # 行走离开精确通约点，周期随振幅漂移。
    assert orbit.period is not None
    assert abs(orbit.period - 4.0 * np.pi) > 0.005


def test_design_ro_default_returns_exact_member() -> None:
    """不指定振幅时返回 3:1 精确成员。"""
    dynamics = CR3BP_Dynamics(earth_moon_system())
    orbit = design_ro(3, 1, dynamics=dynamics)
    assert orbit.period is not None
    assert abs(orbit.period - 2.0 * np.pi) / (2.0 * np.pi) < 1e-9
    measured = _amplitude_km(dynamics, orbit)
    assert abs(measured - _RO_31_ANCHOR_AMPLITUDE_KM) < 100.0


def test_registry_dispatches_ro() -> None:
    """族注册表含 RO 条目，按请求参数形状分发。"""
    assert "RO" in registry
    orbit = registry["RO"](resonance_p=3, resonance_q=1)
    assert orbit.period is not None
    assert abs(orbit.period - 2.0 * np.pi) / (2.0 * np.pi) < 1e-9


def test_design_ro_family_continuation_chain() -> None:
    """RO 族延拓链：窗口内多成员、按振幅升序、相邻成员自然延拓连续。

    选 4:1 族的平缓行走区（振幅随 x0 每步约 20 km）。
    """
    result = design_ro_family(4, 1, 151000.0, 151800.0, n_orbits=4)
    assert result.status is ConvergenceState.CONVERGED, result.message
    family = result.family
    assert family.family_type == "ro"
    members = list(family.orbits)
    assert len(members) >= 3
    amplitudes = [orbit.parameters["amplitude_km"] for orbit in members]
    assert amplitudes == sorted(amplitudes)
    assert all(151000.0 <= amp <= 151800.0 for amp in amplitudes)
    # 相邻成员自然延拓：x0 步长即名义延拓步长（0.004），链上无跳支
    x0s = [float(orbit.states[0, 0]) for orbit in members]
    assert x0s == sorted(x0s)  # 振幅随 x0 单调
    gaps = np.diff(x0s)
    assert np.all(gaps <= 0.005 + 1e-12)
    # 全部成员严格周期闭合
    for orbit in members:
        assert orbit.closure_error is not None and orbit.closure_error < 1e-6
        assert orbit.period is not None
    # 精确共振成员在链上（窗口跨种子振幅）
    assert any(abs(orbit.period - 2.0 * np.pi) < 1e-6 for orbit in members), (
        "族链应包含 4:1 精确共振成员（T = 2πq = 2π）"
    )
    # 成员参数携带共振比 provenance
    assert all(orbit.parameters["resonance_p"] == 4 for orbit in members)


def test_design_ro_family_rejects_unsupported_resonance() -> None:
    with pytest.raises(ValueError, match="不支持的共振比"):
        design_ro_family(1, 2, 151000.0, 151800.0)
