"""TransferSearch.search(parallel_backend='rust') 端到端 + monkeypatch 回退（阶段 D1）。

验证两点：

1. ``search(parallel_backend='rust')`` 端到端跑通，候选解与 ``search(n_workers=1)``
   （Python sequential）逐字段等价——整数索引精确相等、布尔精确相等、浮点 allclose。
   Rust 路径补齐 ``departure_time_index`` / 名称字段，对齐 sequential 后端。
2. 几何方法被 ``monkeypatch.setattr`` 替换时，``search(parallel_backend='rust')``
   自动回退 Python 路径（fake 被调用），不报错——monkeypatch 缝兼容生效。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

# 扩展未构建时（doc build / 无 spice 构建合法）整模块跳过。
pytest.importorskip("e2m2e._integrators")

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.transfer import TransferSearch, search_parallel
from e2m2e.data.constants import Datum
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


MU = Datum.DE421.mu  # 地月质量参数

# 整数索引字段：精确相等（含 None，未命中时两边都 None）。
INT_FIELDS = [
    "min_distance_idx",
    "min_distance_orbit_idx",
    "intersection_idx",
    "first_intersection_idx",
    "first_min_distance_idx",
    "local_minimum_idx",
    "collision_idx",
]

# 布尔字段：精确相等。
BOOL_FIELDS = [
    "success",
    "intersection_found",
    "collision_found",
    "local_minimum_found",
]

# 浮点字段：allclose（含 None / inf 特例处理）。
FLOAT_FIELDS = [
    "min_distance",
    "dv_departure",
    "dv_insertion",
    "transfer_time",
    "local_minimum_distance",
    "alpha",
    "departure_time",
    "first_intersection_time",
    "first_min_distance_time",
]


def _circular_orbit(xc: float, r: float, n: int, t_max: float = 6.28) -> Orbit:
    """构造绕 (xc, 0) 的平面圆轨道（CR3BP 无量纲），n 点等间隔采样。"""
    t = np.linspace(0.0, t_max, n)
    states = np.zeros((n, 6))
    states[:, 0] = xc + r * np.cos(t)
    states[:, 1] = r * np.sin(t)
    states[:, 3] = -r * np.sin(t)
    states[:, 4] = r * np.cos(t)
    return Orbit(states, t)


@pytest.fixture
def system() -> CR3BP_System:
    return CR3BP_System(mu=MU, primary="earth", secondary="moon")


@pytest.fixture
def dynamics(system: CR3BP_System) -> CR3BP_Dynamics:
    d = CR3BP_Dynamics(system)
    d.integrator = "DOP853"
    d.rtol = d.atol = 1e-9
    d.max_step = 0.05
    return d


@pytest.fixture
def searcher(system: CR3BP_System, dynamics: CR3BP_Dynamics) -> TransferSearch:
    s = TransferSearch(dynamics)
    s.alpha_min = 0.9
    s.alpha_max = 1.1
    s.n_alpha = 3
    s.n_departure = 2
    s.max_transfer_time = 0.5
    s.intersection_threshold = 1e-3
    s.min_distance_threshold = 0.05
    s.collision_earth_radius = 5e-4
    s.collision_moon_radius = 3e-4
    s.integration_dt = 0.02
    return s


@pytest.fixture
def dep_orbit() -> Orbit:
    return _circular_orbit(0.9, 0.08, 80)


@pytest.fixture
def arr_orbit() -> Orbit:
    return _circular_orbit(0.7, 0.12, 60)


def _sort_key(r: dict[str, Any]) -> tuple:
    """按 (departure_time, alpha) 排序；两边返回顺序本就一致，排序仅为保险。"""
    return (round(float(r["departure_time"]), 12), float(r["alpha"]))


def _assert_candidate_equal(py_r: dict[str, Any], rust_r: dict[str, Any]) -> None:
    """逐字段对照 Python sequential 与 Rust 后端的单个候选解。"""
    assert py_r["status"] == rust_r["status"], (
        f"status: py={py_r['status']!r} rust={rust_r['status']!r}"
    )
    assert py_r["collision_body"] == rust_r["collision_body"], (
        f"collision_body: py={py_r['collision_body']!r} rust={rust_r['collision_body']!r}"
    )

    for f in INT_FIELDS:
        assert py_r[f] == rust_r[f], f"{f}: py={py_r[f]!r} rust={rust_r[f]!r}"

    for f in BOOL_FIELDS:
        assert py_r[f] == rust_r[f], f"{f}: py={py_r[f]!r} rust={rust_r[f]!r}"

    integration_failed = py_r["status"] == "integration_failed"
    for f in FLOAT_FIELDS:
        py_v = py_r[f]
        rust_v = rust_r[f]
        if py_v is None:
            assert rust_v is None, f"{f}: py=None rust={rust_v!r}"
            continue
        assert rust_v is not None, f"{f}: py={py_v!r} rust=None"
        if np.isinf(py_v) or np.isnan(py_v):
            assert py_v == rust_v, f"{f}: py={py_v!r} rust={rust_v!r}"
            continue
        # 积分发散分支：Rust dv_departure=1e10 惩罚，Python 保留真实值，跳过。
        if integration_failed and f == "dv_departure":
            continue
        assert_allclose(rust_v, py_v, rtol=1e-9, atol=1e-12, err_msg=f"{f}")

    assert_allclose(
        rust_r["departure_state"],
        py_r["departure_state"],
        rtol=1e-12,
        atol=1e-12,
        err_msg="departure_state",
    )

    py_traj = py_r.get("transfer_trajectory")
    rust_traj = rust_r.get("transfer_trajectory")
    if py_traj is None:
        assert rust_traj is None, "transfer_trajectory: py=None rust 非 None"
    else:
        assert rust_traj is not None, "transfer_trajectory: py 非 None rust=None"
        py_traj = np.asarray(py_traj, dtype=float)
        rust_traj = np.asarray(rust_traj, dtype=float)
        assert py_traj.shape == rust_traj.shape, (
            f"transfer_trajectory shape: py={py_traj.shape} rust={rust_traj.shape}"
        )
        assert_allclose(rust_traj, py_traj, rtol=1e-9, atol=1e-12, err_msg="transfer_trajectory")


def test_search_rust_backend_end_to_end(
    searcher: TransferSearch,
    dep_orbit: Orbit,
    arr_orbit: Orbit,
) -> None:
    """search(parallel_backend='rust') 与 search(n_workers=1) 逐候选等价。"""
    results_rust = searcher.search(
        departure_orbit=dep_orbit,
        arrival_orbit=arr_orbit,
        verbose=False,
        n_workers=2,
        parallel_backend="rust",
    )
    results_py = searcher.search(
        departure_orbit=dep_orbit,
        arrival_orbit=arr_orbit,
        verbose=False,
        n_workers=1,
        parallel_backend="processes",
    )

    n_expected = int(searcher.n_departure) * int(searcher.n_alpha)
    assert len(results_rust) == n_expected
    assert len(results_py) == n_expected

    results_rust.sort(key=_sort_key)
    results_py.sort(key=_sort_key)
    for py_r, rust_r in zip(results_py, results_rust, strict=True):
        _assert_candidate_equal(py_r, rust_r)

    # Rust 路径补齐 departure_time_index / 名称字段（对齐 sequential 后端）。
    for r in results_rust:
        assert "departure_time_index" in r
        assert "departure_orbit_name" in r
        assert "arrival_orbit_name" in r


def test_search_rust_falls_back_when_monkeypatched(
    searcher: TransferSearch,
    dep_orbit: Orbit,
    arr_orbit: Orbit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """几何方法被 monkeypatch 时，search(parallel_backend='rust') 回退 Python。

    回退路径经过 fake 函数（Rust 内核不调用 Python 方法），故 fake 调用次数 > 0
    即证明走了 Python 回退而非 Rust。
    """
    call_count = {"n": 0}

    def _fake_integrate(self: TransferSearch, initial_state: np.ndarray, t_end: float, dt: float):
        call_count["n"] += 1
        n = 30
        traj = np.zeros((n, 6))
        # 远离 arrival_orbit（x≈0.7）与天体，确保 no_intersection 且不碰撞。
        traj[:, 0] = 50.0 + np.linspace(0.0, 10.0, n)
        times = np.linspace(0.0, float(t_end), n)
        return traj, times

    monkeypatch.setattr(TransferSearch, "_forward_integrate", _fake_integrate, raising=True)

    results = searcher.search(
        departure_orbit=dep_orbit,
        arrival_orbit=arr_orbit,
        verbose=False,
        n_workers=1,
        parallel_backend="rust",
    )

    # 回退到 sequential，fake 被调用 n_departure * n_alpha 次（每个 α 一次）。
    assert call_count["n"] == int(searcher.n_departure) * int(searcher.n_alpha)
    n_expected = int(searcher.n_departure) * int(searcher.n_alpha)
    assert len(results) == n_expected


def test_search_default_backend_is_rust_when_built(
    searcher: TransferSearch,
    dep_orbit: Orbit,
    arr_orbit: Orbit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """扩展已构建时默认 backend = rust（#316 item ④）。

    TransferSearch() 构造后 ``_parallel_backend`` 由 :func:`_default_parallel_backend`
    决定；扩展已构建（本测试被 importorskip 守卫）→ ``"rust"``。``search()`` 不传
    ``parallel_backend`` 时走实例默认 → 路由到 :func:`grid_search_rust_dispatch`。
    """
    assert searcher._parallel_backend == "rust"

    rust_called = {"v": False}
    real = search_parallel.grid_search_rust_dispatch

    def spy(*args: object, **kwargs: object) -> list[dict[str, Any]]:
        rust_called["v"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(search_parallel, "grid_search_rust_dispatch", spy)

    results = searcher.search(
        departure_orbit=dep_orbit,
        arrival_orbit=arr_orbit,
        verbose=False,
        n_workers=2,
    )
    assert rust_called["v"], "默认 backend 应路由到 grid_search_rust_dispatch（rust）"
    assert len(results) == int(searcher.n_departure) * int(searcher.n_alpha)


def test_set_parallel_backend_then_search_routes_to_it(
    searcher: TransferSearch,
    dep_orbit: Orbit,
    arr_orbit: Orbit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """set_parallel_backend('processes') 后 search() 不传 backend → 走 processes。

    验证 search() 的 ``parallel_backend=None`` sentinel 确实回落到实例属性（此前
    search() 参数默认是 ``"processes"`` 字面量，set_parallel_backend 对它不生效）。
    走 processes 时 dispatch 不应碰 grid_search_rust_dispatch。
    """
    searcher.set_parallel_backend("processes")
    assert searcher._parallel_backend == "processes"

    def rust_should_not_fire(*args: object, **kwargs: object) -> list[dict[str, Any]]:
        raise AssertionError("backend=processes 不应路由到 grid_search_rust_dispatch")

    monkeypatch.setattr(search_parallel, "grid_search_rust_dispatch", rust_should_not_fire)

    results = searcher.search(
        departure_orbit=dep_orbit,
        arrival_orbit=arr_orbit,
        verbose=False,
        n_workers=1,  # processes + n_workers=1 → sequential
    )
    assert len(results) == int(searcher.n_departure) * int(searcher.n_alpha)
