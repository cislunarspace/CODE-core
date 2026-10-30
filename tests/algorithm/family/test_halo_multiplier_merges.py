"""乘子合并定位：分类器单元测试与 L2 南 Halo 族扫描路径。"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family.orbits.walk import earth_moon_system
from e2m2e.algorithm.family.resonant_branch import (
    TRIVIAL_PROJECTION_TOLERANCE,
    classify_multiplier_pairs,
    locate_halo_multiplier_merges,
)
from e2m2e.data.types.orbit import OrbitFamily
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
