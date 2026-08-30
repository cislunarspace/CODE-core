"""catalog_sweep 参数扫描编排测试：批量生成与部分失败保留（ADR 0020/0029/0031）。

能量（Jacobi）窗口点断外部行为：各窗口记录成员的 Jacobi 均在窗口内、
空窗口无记录但结局可查、同（族、平动点）窗口共享一次批量生成调用
（trace 只生成一次的行为契约）。
"""

from __future__ import annotations

import functools

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


def _windowed_halo_point(window: tuple[float, float]) -> FamilySweepPoint:
    return FamilySweepPoint(
        orbit_type="HALO",
        libration_point=1,
        n_orbits=3,
        kwargs={"max_amplitude_km": 3000.0},
        jacobi_window=window,
    )


@functools.cache
def _halo_jacobi_bounds() -> tuple[float, float]:
    """探出 L1 Halo（3000 km）族的真实 Jacobi 包络，供测试自校准窗口。

    模块级缓存：确定性探测只跑一次，多个窗口测试共享、消除重复计算
    （探测本身是一次 n_orbits=6 的族生成）。
    """
    probe = run_family_sweep([_halo_point(3000.0, n_orbits=6)])[0]
    assert probe.result is not None
    jacobis = probe.result.family.get_jacobi_constants()
    return float(jacobis.min()), float(jacobis.max())


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
                    orbit_type="DRO",
                    libration_point=2,
                    n_orbits=1,
                    kwargs={},
                )
            ]
        )
        assert outcomes[0].result is None
        assert outcomes[0].status is ConvergenceState.FAILED
        assert outcomes[0].cause is FailureCause.INVALID_INPUT


class TestLissajousGrid:
    def test_lissajous_grid_point_generates_quasi_periodic_family(self):
        outcomes = run_family_sweep(
            [
                FamilySweepPoint(
                    orbit_type="LISSAJOUS",
                    libration_point=2,
                    n_orbits=2,
                    kwargs={
                        "amplitude_in_km": 1500.0,
                        "amplitude_out_km": 4000.0,
                        "phase_in": 0.01,
                        "phase_out": 0.55,
                    },
                )
            ]
        )

        assert len(outcomes) == 1
        result = outcomes[0].result
        assert outcomes[0].status is ConvergenceState.CONVERGED
        assert result is not None
        assert result.family.family_type == "lissajous"
        assert result.family.is_quasi_periodic
        assert result.generated_members == 2


class TestJacobiWindows:
    def test_windowed_points_keep_members_inside_window(self):
        lower, upper = _halo_jacobi_bounds()
        middle = 0.5 * (lower + upper)
        windows = [(lower, middle), (middle, upper)]

        outcomes = run_family_sweep([_windowed_halo_point(window) for window in windows])

        assert len(outcomes) == 2
        for outcome, (window_lo, window_hi) in zip(outcomes, windows, strict=True):
            assert outcome.status is ConvergenceState.CONVERGED
            assert outcome.result is not None
            member_jacobis = outcome.result.family.get_jacobi_constants()
            assert len(member_jacobis) == outcome.result.generated_members
            assert member_jacobis.min() >= window_lo
            assert member_jacobis.max() <= window_hi

    def test_windowed_axial_points_cover_second_family(self):
        probe = run_family_sweep(
            [
                FamilySweepPoint(
                    orbit_type="AXIAL",
                    libration_point=1,
                    n_orbits=5,
                    kwargs={
                        "max_amplitude_km": 5000.0,
                        "continuation_direction": "increase-amplitude",
                    },
                )
            ]
        )[0]
        assert probe.result is not None
        jacobis = probe.result.family.get_jacobi_constants()
        lower, upper = float(jacobis.min()), float(jacobis.max())
        middle = 0.5 * (lower + upper)
        windows = [(lower, middle), (middle, upper)]

        outcomes = run_family_sweep(
            [
                FamilySweepPoint(
                    orbit_type="AXIAL",
                    libration_point=1,
                    n_orbits=3,
                    kwargs={
                        "max_amplitude_km": 5000.0,
                        "continuation_direction": "increase-amplitude",
                    },
                    jacobi_window=window,
                )
                for window in windows
            ]
        )

        assert len(outcomes) == 2
        for outcome, (window_lo, window_hi) in zip(outcomes, windows, strict=True):
            assert outcome.status is ConvergenceState.CONVERGED
            assert outcome.result is not None
            member_jacobis = outcome.result.family.get_jacobi_constants()
            assert member_jacobis.min() >= window_lo
            assert member_jacobis.max() <= window_hi

    def test_windowed_group_shares_single_generation_call(self, monkeypatch):
        import e2m2e.algorithm.catalog_sweep as sweep_module
        from e2m2e.algorithm.results import FamilyGenerationResult
        from e2m2e.data.types.orbit import OrbitFamily

        calls: list[tuple[str, int, list]] = []

        # ADR 0037 预算内：批量生成打桩为合成结果。本测试只验证"同组窗口点共享
        # 一次批量调用"的编排契约，与真实生成内容无关；真实窗口筛选行为由
        # test_windowed_points_keep_members_inside_window 等承担。
        def stub(family_type, libration_point, n_orbits, windows, **kwargs):
            calls.append((family_type, libration_point, list(windows)))
            return [
                FamilyGenerationResult(
                    status=ConvergenceState.CONVERGED,
                    cause=FailureCause.NONE,
                    message="测试完成",
                    family=OrbitFamily(family_type=family_type),
                    requested_members=n_orbits,
                    generated_members=0,
                )
                for _ in windows
            ]

        monkeypatch.setattr(sweep_module, "generate_rust_family_windows", stub)
        # 自包含的任意窗口（不依赖 _halo_jacobi_bounds 真实探测）。
        lower, middle, upper = 3.0, 3.1, 3.2

        outcomes = run_family_sweep(
            [
                _halo_point(2000.0),  # 非窗口点走逐点路径，不进组
                _windowed_halo_point((lower, middle)),
                _windowed_halo_point((middle, upper)),
                _windowed_halo_point((lower, upper)),
            ]
        )

        # 同（族、平动点、生成参数）的三个窗口点共享一次批量生成调用
        assert len(calls) == 1
        assert calls[0][0] == "halo"
        assert calls[0][2] == [(lower, middle), (middle, upper), (lower, upper)]
        assert len(outcomes) == 4
        assert outcomes[0].point.jacobi_window is None
        assert [outcome.point.jacobi_window for outcome in outcomes[1:]] == [
            (lower, middle),
            (middle, upper),
            (lower, upper),
        ]

    def test_empty_window_is_soft_failure_without_record_payload(self):
        outcomes = run_family_sweep([_windowed_halo_point((9.9, 9.95))])

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.result is not None
        assert outcome.status is ConvergenceState.INFEASIBLE
        assert outcome.cause is FailureCause.CONSTRAINT_VIOLATION
        assert outcome.result.generated_members == 0
        assert len(outcome.result.family) == 0
        assert "零成员" in outcome.message

    def test_windowed_generation_failure_keeps_produced_results(self, monkeypatch):
        """一组窗口点硬失败时，另一组的已产结果保留（ADR 0020 软失败语义）。"""
        import e2m2e.algorithm.catalog_sweep as sweep_module

        real = sweep_module.generate_rust_family_windows
        lower, upper = _halo_jacobi_bounds()

        def boom_on_wide_extent(family_type, libration_point, n_orbits, windows, **kwargs):
            if kwargs["max_amplitude_km"] > 5000.0:
                raise RuntimeError("Rust 批量生成崩溃")
            return real(family_type, libration_point, n_orbits, windows, **kwargs)

        monkeypatch.setattr(sweep_module, "generate_rust_family_windows", boom_on_wide_extent)
        outcomes = run_family_sweep(
            [
                FamilySweepPoint(
                    orbit_type="HALO",
                    libration_point=1,
                    n_orbits=2,
                    kwargs={"max_amplitude_km": 2000.0},
                    jacobi_window=(lower, upper),
                ),
                FamilySweepPoint(
                    orbit_type="HALO",
                    libration_point=1,
                    n_orbits=2,
                    kwargs={"max_amplitude_km": 8000.0},
                    jacobi_window=(lower, upper),
                ),
            ]
        )

        assert [outcome.result is not None for outcome in outcomes] == [True, False]
        produced = outcomes[0].result
        assert produced is not None
        assert produced.generated_members >= 1
        failed = outcomes[1]
        assert failed.status is ConvergenceState.FAILED
        assert "崩溃" in failed.message
