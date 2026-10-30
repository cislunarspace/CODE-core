"""乘子合并定位：分类器单元测试与 L2 南 Halo 族扫描路径。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family.orbits.walk import Cr3bpOrbitError, earth_moon_system
from e2m2e.algorithm.family.resonant_branch import (
    CRITICAL_MULTIPLIER_TOLERANCE,
    TRIVIAL_PROJECTION_TOLERANCE,
    MultiplierMerge,
    _is_startable,
    _prepare_halo_seed,
    _resonance_from_merge,
    _tight_halo,
    classify_multiplier_pairs,
    locate_halo_multiplier_merges,
)
from e2m2e.data.types.orbit import Orbit, OrbitFamily
from e2m2e.status import ConvergenceState, FailureCause


def _synthetic_monodromy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造谱已知的实 6×6 矩阵：一平凡对、一实倒数非平凡对、一单位圆复数对。"""
    flow = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tangent = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    transverse = 0.5 * flow + np.sqrt(0.75) * np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    reciprocal = 0.5 * flow + np.sqrt(0.75) * np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    rotation = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) + 1j * np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    eigenvalues = np.array([1.0 + 1e-6, 1.0 - 1e-6, 2.0, 0.5, 1j, -1j])
    eigenvectors = np.column_stack(
        [flow, tangent, transverse, reciprocal, rotation, np.conj(rotation)]
    )
    matrix = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
    return np.real(matrix), flow, tangent


def _synthetic_merge(
    *, refined: bool, gap: float = 1e-4, projection: float = 0.1
) -> MultiplierMerge:
    """构造一个 +1 合并事件，只暴露起链判定关心的三个量。"""
    return MultiplierMerge(
        kind="+1",
        parameter=1.082868,
        member_index=138,
        state=np.zeros(6),
        period=2.382171,
        multiplier=1.0 + gap,
        multiplier_gap=gap,
        nu=2.0,
        eigenvector=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        flow_projection=projection,
        tangent_projection=projection,
        bracket=(1.082349, 1.082935),
        refined=refined,
    )


@pytest.mark.theory
def test_startable_requires_refined_gap_and_projection() -> None:
    """起链判定同时要求方向可分、精化达标与乘子偏差在拒绝门限内。"""
    assert _is_startable(_synthetic_merge(refined=True))
    assert not _is_startable(_synthetic_merge(refined=False))
    assert not _is_startable(
        _synthetic_merge(refined=True, gap=2.0 * CRITICAL_MULTIPLIER_TOLERANCE)
    )
    assert not _is_startable(
        _synthetic_merge(refined=True, projection=TRIVIAL_PROJECTION_TOLERANCE)
    )


@pytest.mark.theory
def test_resonance_from_merge_maps_starting_point() -> None:
    """起链事件转共振点：单圈、零相位，状态与周期原样带入，复向量取实部。"""
    event = _synthetic_merge(refined=True)
    resonance = _resonance_from_merge(event)

    assert resonance.revolutions == 1
    assert resonance.phase_deg == pytest.approx(0.0)
    assert resonance.x0 == pytest.approx(event.parameter)
    assert resonance.period == pytest.approx(event.period)
    assert resonance.multiplier_gap == pytest.approx(event.multiplier_gap)
    np.testing.assert_allclose(resonance.state, event.state)

    complex_event = replace(event, eigenvector=np.array([1.0, 1.0j, 0.0, 0.0, 0.0, 0.0]))
    assert np.isrealobj(_resonance_from_merge(complex_event).eigenvector)


@pytest.mark.theory
def test_tight_halo_reports_failure(
    monkeypatch: pytest.MonkeyPatch, dynamics: CR3BP_Dynamics
) -> None:
    """收紧失败时返回原成员并标记 False，端点不会被记成已达到精化容差。"""
    orbit = Orbit(states=np.zeros((1, 6)), times=np.zeros(1), system=dynamics.system)
    orbit.period = 2.4

    def fail(*args: object, **kwargs: object) -> None:
        raise Cr3bpOrbitError("固定 x0 修正未收敛")

    monkeypatch.setattr("e2m2e.algorithm.family.resonant_branch._correct_halo_x0", fail)
    tightened_orbit, tightened = _tight_halo(dynamics, orbit, None)

    assert tightened is False
    assert tightened_orbit is orbit


@pytest.mark.theory
def test_seed_with_non_l2_libration_point_is_rejected(dynamics: CR3BP_Dynamics) -> None:
    """调用方给 L1 族成员时显式拒绝，不静默按 L2 南族标注。"""
    seed = Orbit(states=np.zeros((1, 6)), times=np.zeros(1), system=dynamics.system)
    seed.period = 2.4
    seed.parameters = {"libration_point": 1}

    with pytest.raises(ValueError, match="平动点必须为 L2"):
        _prepare_halo_seed(dynamics, seed)


@pytest.mark.theory
def test_classifier_separates_trivial_pair_by_projection() -> None:
    matrix, flow, tangent = _synthetic_monodromy()
    pairs = classify_multiplier_pairs(matrix, flow, tangent)

    assert len(pairs) == 3
    trivial = [pair for pair in pairs if pair.is_trivial]
    assert len(trivial) == 1
    assert trivial[0].nu == pytest.approx(2.0, abs=1e-9)
    assert trivial[0].flow_projection > TRIVIAL_PROJECTION_TOLERANCE
    assert trivial[0].tangent_projection > TRIVIAL_PROJECTION_TOLERANCE

    nontrivial = [pair for pair in pairs if not pair.is_trivial]
    real_pair = next(pair for pair in nontrivial if abs(pair.nu - 2.5) < 1e-9)
    assert real_pair.flow_projection == pytest.approx(0.5, abs=1e-9)
    assert real_pair.tangent_projection < 1e-9
    assert any(abs(pair.nu) < 1e-9 for pair in nontrivial)


@pytest.fixture(scope="module")
def dynamics() -> CR3BP_Dynamics:
    return CR3BP_Dynamics(earth_moon_system())


@pytest.fixture(scope="module")
def scan(dynamics: CR3BP_Dynamics):
    return locate_halo_multiplier_merges(dynamics, max_members=142)


@pytest.mark.orchestration
def test_scan_records_minus_one_merge(scan) -> None:
    """行走到周期倍增点，记录乘子过 −1 的合并并精化到 ν 贴近 −2。"""
    minus = [event for event in scan.events if event.kind == "-1"]
    assert minus
    for event in minus:
        assert event.nu == pytest.approx(-2.0, abs=1e-3)
        assert event.multiplier_gap < 1e-2
        assert event.bracket[0] <= event.bracket[1]
    assert any(event.refined for event in minus)


@pytest.mark.orchestration
def test_scan_reports_plus_one_merge_without_transverse_direction(scan) -> None:
    """+1 合并被如实记录，但临界特征向量退化为流方向或族切向，不起链。"""
    plus = [event for event in scan.events if event.kind == "+1"]
    assert plus
    for event in plus:
        assert event.nu == pytest.approx(2.0, abs=1e-2)
        assert event.multiplier_gap < 5e-2
        assert max(event.flow_projection, event.tangent_projection) >= TRIVIAL_PROJECTION_TOLERANCE
    assert scan.resonance is None
    assert scan.status is ConvergenceState.INFEASIBLE
    assert scan.cause is FailureCause.NO_INTERSECTION
    assert "横截特征向量" in scan.message
    assert "最小偏差" in scan.message


@pytest.mark.orchestration
def test_scan_without_crossing_reports_interval_and_gap(dynamics: CR3BP_Dynamics) -> None:
    """成员数不足时如实报告扫描区间与最小乘子偏差，不抛异常。"""
    scan = locate_halo_multiplier_merges(dynamics, max_members=2)
    assert scan.status is ConvergenceState.INFEASIBLE
    assert scan.cause is FailureCause.NO_INTERSECTION
    assert scan.events == ()
    assert scan.resonance is None
    assert scan.members_scanned >= 2
    assert "x0 区间" in scan.message
    assert "最小偏差" in scan.message


@pytest.mark.orchestration
def test_scan_reports_walk_without_pairs(
    monkeypatch: pytest.MonkeyPatch, dynamics: CR3BP_Dynamics
) -> None:
    """PAL 只返回种子成员时如实报告行走中断，不因零族切向抛异常。"""

    def seed_only(continuation, seed_orbit, **kwargs):
        return OrbitFamily([seed_orbit])

    monkeypatch.setattr(
        "e2m2e.algorithm.family.resonant_branch.halo_pseudo_arclength_continuation", seed_only
    )
    scan = locate_halo_multiplier_merges(dynamics, max_members=2)
    assert scan.status is ConvergenceState.INFEASIBLE
    assert scan.cause is FailureCause.NO_INTERSECTION
    assert scan.events == ()
    assert scan.resonance is None
    assert scan.members_scanned == 1
    assert scan.trace == ()
    assert "提前中断" in scan.message
