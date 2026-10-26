"""周期倍化分岔定位与两族生成的行为测试（#821）。

定位与族生成走真实链路（母族行走、单值矩阵传播、deflection 起链与
PAL 行走），定位点按模块缓存。验收全部定义性：三判据数值进门限、定位
位置复现、闭合残差、成员数含种子不超请求上限、族容器与成员参数契约。
文献数值（Grebow 表 3.13）不进断言，逐行对照在
``scripts/compare_grebow_table_3_13.py``。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family.orbits.period_doubling import (
    BROUCKE_F_TOLERANCE,
    MULTIPLIER_GAP_TOLERANCE,
    REDUCED_SIGMA_TOLERANCE,
    design_figure_eight_family,
    design_ring_family,
    locate_period_doubling_bifurcations,
)
from e2m2e.algorithm.family.orbits.walk import earth_moon_system
from e2m2e.algorithm.orbit_taxonomy import classify_orbit
from e2m2e.data.constants import EARTH_MOON_DISTANCE_KM

pytestmark = pytest.mark.orchestration


@pytest.fixture(scope="module")
def dynamics():
    return CR3BP_Dynamics(earth_moon_system())


@pytest.fixture(scope="module")
def points(dynamics):
    return locate_period_doubling_bifurcations(dynamics)


def test_locator_finds_four_crossings_with_consistent_criteria(points):
    """该扫描段四处 f 过零，每个定位点三判据同时进门限。"""
    assert len(points) == 4
    for point in points:
        assert abs(point.broucke_f) < BROUCKE_F_TOLERANCE
        assert abs(point.reduced_sigma) < REDUCED_SIGMA_TOLERANCE
        assert point.multiplier_gap < MULTIPLIER_GAP_TOLERANCE
        assert float(point.state[1]) == pytest.approx(0.0, abs=1e-12)


def test_defined_bifurcations_match_reference_positions(points):
    """0 形族分岔与 issue 复算值、8 字形族分岔与库内定位复现值均在
    1e-6 内一致（定位复现性锚点）。文献四位小数值 1.0118 的一致性
    对照在 scripts/compare_grebow_table_3_13.py（ADR 0055：结果性
    文献值不进断言）。"""
    ring = min(points, key=lambda p: abs(p.x0 - 1.109352298816))
    assert abs(ring.x0 - 1.109352298816) < 1e-6
    figure_eight = min(points, key=lambda p: abs(p.x0 - 1.0118))
    assert abs(figure_eight.x0 - 1.011828910724) < 1e-6


def test_invalid_arguments_are_rejected(dynamics):
    """入口壳的参数校验先于计算拒绝非法取值。"""
    with pytest.raises(ValueError, match="仅支持 L2"):
        design_figure_eight_family(1, 1, 388900.0, 411600.0, dynamics=dynamics)
    with pytest.raises(ValueError, match="north_south"):
        design_ring_family(2, 3, 426400.0, 453500.0, dynamics=dynamics)
    with pytest.raises(ValueError, match="continuation_direction"):
        design_ring_family(
            2, 1, 426400.0, 453500.0, continuation_direction="sideways", dynamics=dynamics
        )
    with pytest.raises(ValueError, match="min_x0_km"):
        design_figure_eight_family(2, 1, 400000.0, 390000.0, dynamics=dynamics)


def test_figure_eight_family_members_close_and_cap(dynamics):
    """8 字形族窄窗生成：含种子的成员数不超请求上限，闭合残差达标，
    成员携带族名与起链参数，实测分类为 halo 北标签（族名与几何标签
    并存的口径，见模块 docstring）。"""
    result = design_figure_eight_family(
        2,
        1,
        1.019 * EARTH_MOON_DISTANCE_KM,
        1.025 * EARTH_MOON_DISTANCE_KM,
        n_orbits=3,
        dynamics=dynamics,
    )
    assert result.status.value == "converged", result.message
    family = result.family
    assert family.family_type == "figure_eight"
    assert len(family) <= 3
    assert result.generated_members == len(family)
    assert result.requested_members == 3
    for orbit in family.orbits:
        assert orbit.closure_error < 1e-9
        assert orbit.family_type == "figure_eight"
        assert orbit.parameters["libration_point"] == 2
        assert orbit.parameters["north_south"] == 1
        primary = classify_orbit(
            orbit.states, orbit.times, period=orbit.period, mu=orbit.system.mu
        ).primary
        assert primary is None or primary.canonical == "halo_l2_northern"
    assert family.metadata["bifurcation_seeding"] == "period-doubling-deflection"
    assert abs(family.metadata["bifurcation_x0"] - 1.011828910724) < 1e-6


def test_ring_family_members_close_and_classify_as_geometry_labels(dynamics):
    """0 形族窄窗生成（南族）：闭合残差达标；有限振幅成员实测分类为
    halo 南或 axial 标签而非 butterfly（每周期 2 次垂直加 2 次斜穿越，
    分类判据不改，#821 与 ADR 0060 修订的口径记录）。"""
    result = design_ring_family(
        2,
        2,
        1.110 * EARTH_MOON_DISTANCE_KM,
        1.113 * EARTH_MOON_DISTANCE_KM,
        n_orbits=2,
        dynamics=dynamics,
    )
    assert result.status.value == "converged", result.message
    family = result.family
    assert family.family_type == "ring"
    assert len(family) <= 2
    for orbit in family.orbits:
        assert orbit.closure_error < 1e-9
        primary = classify_orbit(
            orbit.states, orbit.times, period=orbit.period, mu=orbit.system.mu
        ).primary
        assert primary is None or primary.canonical in {"halo_l2_southern", "axial_l2"}
    assert abs(family.metadata["bifurcation_x0"] - 1.109352298816) < 1e-6


def test_window_above_seed_collects_only_in_range_members(dynamics):
    """窗口下界高于种子 x0 时（评审修复回归）：链穿过窗口，收集落入
    窗口的成员，种子保留、窗口外成员不收集。"""
    result = design_figure_eight_family(
        2,
        1,
        400000.0,
        405000.0,
        n_orbits=8,
        dynamics=dynamics,
    )
    assert result.status.value == "converged", result.message
    xs = [float(o.states[0, 0]) * EARTH_MOON_DISTANCE_KM for o in result.family.orbits]
    assert len(xs) > 2  # 只有种子说明链没穿过窗口（旧缺陷症状）
    assert all(400000.0 <= x <= 405000.0 for x in xs[1:])
