"""catalog_sweep 参数扫描编排测试：批量生成与部分失败保留（ADR 0020/0029/0031）。"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.catalog_sweep import (
    FamilySweepPoint,
    run_family_sweep,
)
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.orchestration


def _halo_point(max_amplitude_km: float, n_orbits: int = 2) -> FamilySweepPoint:
    return FamilySweepPoint(
        orbit_type="HALO",
        libration_point=1,
        n_orbits=n_orbits,
        kwargs={"max_amplitude_km": max_amplitude_km},
    )


class TestFamilySweep:
    def test_all_points_generate(self):
        outcomes = run_family_sweep([_halo_point(2000.0), _halo_point(3000.0)])

        assert len(outcomes) == 2
        for outcome in outcomes:
            assert outcome.status is ConvergenceState.CONVERGED
            assert outcome.cause is FailureCause.NONE
            assert outcome.result is not None
            assert outcome.result.generated_members >= 1
            assert outcome.result.family.family_type == "halo"

    def test_failed_point_preserves_produced_results(self):
        outcomes = run_family_sweep(
            [
                _halo_point(2000.0),
                _halo_point(0.0),  # 算法层拒绝（振幅不能为 0）
                _halo_point(3000.0),
            ]
        )

        assert [outcome.result is not None for outcome in outcomes] == [True, False, True]
        failed = outcomes[1]
        assert failed.status is ConvergenceState.FAILED
        assert failed.cause is not FailureCause.NONE
        assert "0" in failed.message
        # 已产出点的结果原样保留，顺序与输入一致
        assert outcomes[0].point.kwargs["max_amplitude_km"] == 2000.0
        assert outcomes[2].point.kwargs["max_amplitude_km"] == 3000.0

    def test_unsupported_family_is_marked_not_raised(self):
        outcomes = run_family_sweep(
            [
                FamilySweepPoint(
                    orbit_type="LISSAJOUS",
                    libration_point=2,
                    n_orbits=1,
                    kwargs={},
                )
            ]
        )
        assert outcomes[0].result is None
        assert outcomes[0].status is ConvergenceState.FAILED
        assert outcomes[0].cause is FailureCause.INVALID_INPUT
