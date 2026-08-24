"""WSB Rust 后端的公开契约测试。"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("e2m2e._integrators")

from e2m2e.algorithm.dynamics import BCR4BPSystem
from e2m2e.algorithm.transfer.wsb import WsbSearchParams, search_wsb_trajectories
from e2m2e.data.constants import Datum
from e2m2e.exceptions import RustExtensionUnavailableError

pytestmark = pytest.mark.orchestration


DU = 384405.0


def _departure_state(system: BCR4BPSystem) -> np.ndarray:
    from e2m2e.algorithm.transfer.hohmann import MU_EARTH, R_EARTH

    r_park = R_EARTH + 200.0
    v_circ = math.sqrt(MU_EARTH / r_park)
    return system.physical_to_dimensionless(np.array([r_park, 0.0, 0.0, 0.0, v_circ, 0.0]))


def _target_state(system: BCR4BPSystem) -> np.ndarray:
    r_target_du = 2000.0 / DU
    return np.array(
        [
            1.0 - system.mu + r_target_du,
            0.0,
            0.0,
            0.0,
            math.sqrt(system.mu / r_target_du),
            0.0,
        ]
    )


def _params() -> WsbSearchParams:
    """小网格放宽物理筛选，稳定产生可比较的搜索结果。"""
    return WsbSearchParams(
        sun_phase_range=(0.0, math.pi),
        n_sun_phase=2,
        departure_phase_range=(0.0, math.pi),
        n_departure_phase=3,
        tof_range=(1.0, 10.0),
        n_tof=2,
        perilune_alt_min=-1_000_000.0,
        perilune_alt_max=1_000_000.0,
        max_total_dv=1_000_000.0,
        h2_energy_threshold=1_000_000.0,
        n_propagation_samples=100,
        # 网格筛选级容差（ADR 0021 修订 #420：默认测试时间上界靠缩小问题
        # 规模保证）。Python 参照无步数截断，研究级 1e-12 会让擦月组合
        # 烧到积分器自适应上限，单组合秒级；1e-10 对 1e-8 断言精度足够。
        rtol=1e-10,
        atol=1e-10,
    )


def _assert_results_equal(expected, actual) -> None:
    assert actual.status is expected.status
    assert actual.cause is expected.cause
    assert len(actual) == len(expected)
    for expected_candidate, actual_candidate in zip(expected, actual, strict=True):
        for name in (
            "sun_phase0",
            "departure_phase",
            "tof_sec",
            "perilune_alt_km",
            "perilune_time_dim",
            "h2_kepler",
            "dv_departure",
            "dv_arrival",
            "total_dv",
            "arrival_time_dim",
        ):
            assert_allclose(
                getattr(actual_candidate, name),
                getattr(expected_candidate, name),
                rtol=1e-8,
                atol=1e-12,
                err_msg=name,
            )
        assert_allclose(actual_candidate.departure_state, expected_candidate.departure_state)
        assert_allclose(actual_candidate.perilune_state, expected_candidate.perilune_state)
        assert_allclose(actual_candidate.arrival_state, expected_candidate.arrival_state)


def test_wsb_rust_serial_matches_explicit_python_reference() -> None:
    """固定小网格上 Rust 串行逐候选等价于显式 Python 参照。"""
    system = BCR4BPSystem.earth_moon(mu=Datum.DE421.mu)
    departure = _departure_state(system)
    target = _target_state(system)
    params = _params()

    rust_result = search_wsb_trajectories(
        departure,
        target,
        system,
        params,
        backend="rust",
        parallel=False,
    )

    _assert_results_equal(_python_reference_result(), rust_result)


@functools.cache
def _python_reference_result():
    """标准太阳参数下的 Python 参照结果（module 级缓存，消除重复计算）。"""
    system = BCR4BPSystem.earth_moon(mu=Datum.DE421.mu)
    return search_wsb_trajectories(
        _departure_state(system),
        _target_state(system),
        system,
        _params(),
        backend="python",
        parallel=False,
    )


def test_wsb_rust_matches_python_with_custom_solar_parameters() -> None:
    """显式 Python 参照继承调用方自定义的 BCR4BP 太阳参数。"""
    system = BCR4BPSystem.earth_moon(mu=Datum.DE421.mu)
    system.sun_mass *= 0.9
    system.sun_distance *= 1.1
    system.sun_angular_rate *= 0.95
    departure = _departure_state(system)
    target = _target_state(system)
    params = _params()

    python_result = search_wsb_trajectories(
        departure,
        target,
        system,
        params,
        backend="python",
        parallel=False,
    )
    rust_result = search_wsb_trajectories(
        departure,
        target,
        system,
        params,
        backend="rust",
        parallel=False,
    )

    _assert_results_equal(python_result, rust_result)


def test_wsb_rayon_parallel_matches_rust_serial() -> None:
    """Rayon 并行不改变 WSB 候选顺序、数值或最终状态。"""
    system = BCR4BPSystem.earth_moon(mu=Datum.DE421.mu)
    departure = _departure_state(system)
    target = _target_state(system)
    params = _params()

    serial_result = search_wsb_trajectories(
        departure,
        target,
        system,
        params,
        parallel=False,
    )
    parallel_result = search_wsb_trajectories(
        departure,
        target,
        system,
        params,
        parallel=True,
        n_workers=2,
    )

    _assert_results_equal(serial_result, parallel_result)


def test_wsb_one_worker_and_environment_serial_match_explicit_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rayon 单 worker 与环境变量强制串行均保留 WSB 候选契约。"""
    system = BCR4BPSystem.earth_moon(mu=Datum.DE421.mu)
    departure = _departure_state(system)
    target = _target_state(system)
    params = _params()

    serial_result = search_wsb_trajectories(departure, target, system, params, parallel=False)
    one_worker_result = search_wsb_trajectories(
        departure,
        target,
        system,
        params,
        parallel=True,
        n_workers=1,
    )
    monkeypatch.setenv("E2M2E_WSB_PARALLEL", "0")
    environment_serial_result = search_wsb_trajectories(departure, target, system, params)

    _assert_results_equal(serial_result, one_worker_result)
    _assert_results_equal(serial_result, environment_serial_result)


def test_wsb_rust_progress_reports_each_sun_phase_tof_task() -> None:
    """WSB Rust 后端按 (sun_phase, tof) 任务汇报可聚合进度。"""
    system = BCR4BPSystem.earth_moon(mu=Datum.DE421.mu)
    params = _params()
    deltas: list[int] = []

    search_wsb_trajectories(
        _departure_state(system),
        _target_state(system),
        system,
        params,
        parallel=True,
        n_workers=2,
        progress_callback=deltas.append,
    )

    assert deltas
    assert all(delta > 0 for delta in deltas)
    assert sum(deltas) == params.n_sun_phase * params.n_tof


def test_wsb_default_rust_failure_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认 Rust 入口的扩展错误直接上抛，绝不偷偷改跑 Python。"""
    import e2m2e.integrators as integrators

    system = BCR4BPSystem.earth_moon(mu=Datum.DE421.mu)

    def unavailable(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RustExtensionUnavailableError("模拟 Rust 扩展缺失")

    monkeypatch.setattr(integrators, "wsb_search_rust", unavailable)
    with pytest.raises(RustExtensionUnavailableError, match="模拟 Rust 扩展缺失"):
        search_wsb_trajectories(
            _departure_state(system),
            _target_state(system),
            system,
            _params(),
            parallel=False,
        )
