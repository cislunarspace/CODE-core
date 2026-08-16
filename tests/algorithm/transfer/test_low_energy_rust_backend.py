"""低能流形配对 Rust 后端的公开契约测试。"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("e2m2e._integrators")

from e2m2e.algorithm.manifold import ManifoldKind, ManifoldTube, PoincareSection
from e2m2e.algorithm.transfer import patch_manifolds
from e2m2e.data.types.orbit import Orbit
from e2m2e.exceptions import RustExtensionUnavailableError

pytestmark = pytest.mark.orchestration


def _arc_at_crossing(state: np.ndarray) -> Orbit:
    """构造在 x=0 平面穿越给定状态的一条两点流形弧。"""
    before = state.copy()
    after = state.copy()
    before[0] = -1.0
    after[0] = 1.0
    return Orbit(states=np.array([before, after]), times=np.array([0.0, 1.0]))


def _tube(states: list[list[float]]) -> ManifoldTube:
    """构造仅供截面配对的合成流形管。"""
    return ManifoldTube(
        orbit=Orbit(states=np.zeros((1, 6)), times=np.array([0.0])),
        kind=ManifoldKind.UNSTABLE,
        branch="+",
        epsilon=1e-4,
        trajectories=[_arc_at_crossing(np.asarray(state, dtype=float)) for state in states],
    )


def _assert_candidates_equal(expected, actual) -> None:
    assert [(c.i_a, c.i_b) for c in actual] == [(c.i_a, c.i_b) for c in expected]
    for expected_candidate, actual_candidate in zip(expected, actual, strict=True):
        assert_allclose(actual_candidate.state_a, expected_candidate.state_a, rtol=0.0, atol=0.0)
        assert_allclose(actual_candidate.state_b, expected_candidate.state_b, rtol=0.0, atol=0.0)
        assert actual_candidate.delta_r == expected_candidate.delta_r
        assert actual_candidate.delta_v == expected_candidate.delta_v
        assert actual_candidate.cost == expected_candidate.cost


def test_patch_manifolds_rust_matches_explicit_python_and_preserves_ties() -> None:
    """Rust 配对保留候选字段与 Python 嵌套循环的并列代价顺序。"""
    section = PoincareSection.plane(axis=0, value=0.0)
    tube_a = _tube([[0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0]])
    tube_b = _tube([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

    python_candidates = patch_manifolds(tube_a, tube_b, section, backend="python")
    rust_candidates = patch_manifolds(
        tube_a,
        tube_b,
        section,
        backend="rust",
        parallel=False,
    )
    default_candidates = patch_manifolds(tube_a, tube_b, section, parallel=False)

    assert [(c.i_a, c.i_b) for c in python_candidates] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    _assert_candidates_equal(python_candidates, rust_candidates)
    _assert_candidates_equal(rust_candidates, default_candidates)


def test_patch_manifolds_default_rust_failure_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认 Rust 配对的扩展错误直接上抛，绝不偷偷改跑 Python。"""
    import e2m2e.integrators as integrators

    def unavailable(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RustExtensionUnavailableError("模拟 Rust 扩展缺失")

    monkeypatch.setattr(integrators, "low_energy_patch_rust", unavailable)
    section = PoincareSection.plane(axis=0, value=0.0)
    tube = _tube([[0, 0, 0, 0, 0, 0]])
    with pytest.raises(RustExtensionUnavailableError, match="模拟 Rust 扩展缺失"):
        patch_manifolds(tube, tube, section)


def test_patch_manifolds_progress_workers_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """配对进度按候选计数，单 worker 与环境串行开关保持等价。"""
    section = PoincareSection.plane(axis=0, value=0.0)
    tube_a = _tube([[0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 1, 0]])
    tube_b = _tube([[0, 1, 0, 0, 1, 0], [0, 3, 0, 0, 0, 0]])
    deltas: list[int] = []

    one_worker = patch_manifolds(
        tube_a,
        tube_b,
        section,
        parallel=True,
        n_workers=1,
        progress_callback=deltas.append,
    )
    monkeypatch.setenv("E2M2E_LOW_ENERGY_PARALLEL", "0")
    environment_serial = patch_manifolds(tube_a, tube_b, section)

    assert sum(deltas) == len(one_worker) == 4
    _assert_candidates_equal(one_worker, environment_serial)


def test_patch_manifolds_rayon_parallel_matches_rust_serial() -> None:
    """Rayon 并行不改变低能配对候选的顺序或数值。"""
    section = PoincareSection.plane(axis=0, value=0.0)
    tube_a = _tube([[0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 1, 0]])
    tube_b = _tube([[0, 1, 0, 0, 1, 0], [0, 3, 0, 0, 0, 0]])

    serial = patch_manifolds(tube_a, tube_b, section, backend="rust", parallel=False)
    parallel = patch_manifolds(tube_a, tube_b, section, backend="rust", parallel=True, n_workers=2)

    _assert_candidates_equal(serial, parallel)
