"""Rust 网格搜索进度回调、n_workers 转发、非 success 轨迹过滤（#316 前三项）。

三个独立档：

1. ``progress_callback`` 出发粒度触发——Rust 端每个 departure 完成调一次回调，
   累加 delta 之和等于出发点数（而非 n_dep*n_alpha）。
2. ``n_workers`` 转发 Rayon ThreadPoolBuilder 后结果与全局池一致——
   evaluate_point 是纯函数，线程数不影响数值与索引。
3. 非 success 候选不回传轨迹——collision / no_intersection 的
   ``transfer_trajectory`` / ``transfer_times`` 为 None（与 Rust evaluate_point 一致）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# 扩展未构建时（doc build / 无 spice 构建合法）整模块跳过。
pytest.importorskip("e2m2e._integrators")

from e2m2e.data.constants import Datum
from e2m2e.integrators import grid_search_rust  # noqa: E402

pytestmark = pytest.mark.orchestration


MU = Datum.DE421.mu  # 地月质量参数


def _circular_orbit(xc: float, r: float, n: int, t_max: float = 6.28) -> np.ndarray:
    """绕 (xc, 0) 的平面圆轨道（CR3BP 无量纲），返回 (n, 6) 状态数组。

    速度取逆时针圆周切向，α 缩放切向速度后轨道仍稳定，不触发积分发散。
    """
    t = np.linspace(0.0, t_max, n)
    states = np.zeros((n, 6))
    states[:, 0] = xc + r * np.cos(t)
    states[:, 1] = r * np.sin(t)
    states[:, 3] = -r * np.sin(t)
    states[:, 4] = r * np.cos(t)
    return states


def _kwargs(
    dep_states: np.ndarray,
    dep_times: np.ndarray,
    alpha_grid: np.ndarray,
    arrival_states: np.ndarray,
    min_distance_threshold: float = 0.05,
) -> dict[str, Any]:
    """组装 grid_search_rust 的标量配置包（与 searcher fixture 默认值一致）。"""
    return {
        "dep_states": dep_states,
        "dep_times": dep_times,
        "alpha_grid": alpha_grid,
        "arrival_states": arrival_states,
        "mu": MU,
        "max_transfer_time": 0.5,
        "integration_dt": 0.02,
        "intersection_threshold": 1e-3,
        "min_distance_threshold": min_distance_threshold,
        "collision_earth_radius": 5e-4,
        "collision_moon_radius": 3e-4,
        "rtol": 1e-9,
        "atol": 1e-9,
        "max_step": 0.05,
    }


def test_progress_callback_fires_per_departure() -> None:
    """progress_callback 出发粒度：sum(delta) == n_dep，len(results) == n_dep*n_alpha。"""
    n_dep, n_alpha = 3, 4
    dep_states = _circular_orbit(0.9, 0.08, 40)[:n_dep]
    dep_times = np.linspace(0.0, 6.28, 40)[:n_dep]
    arrival_states = _circular_orbit(0.7, 0.12, 30)
    alpha_grid = np.linspace(0.9, 1.05, n_alpha)

    deltas: list[int] = []

    def cb(delta: int) -> None:
        deltas.append(delta)

    results = grid_search_rust(
        **_kwargs(dep_states, dep_times, alpha_grid, arrival_states),
        n_workers=2,
        progress_callback=cb,
    )
    assert len(results) == n_dep * n_alpha
    # 出发粒度：每个 departure 最后一个 α 完成时 send 一次（=1），drainer 可能聚合
    # 多次 send 为一次 call1（降 GIL 开销），故 len(deltas) ≤ n_dep；但 delta 恒 > 0
    # （completed 单调不减）、sum == n_dep（终值==出发点数）。
    assert sum(deltas) == n_dep
    assert len(deltas) <= n_dep
    assert all(d > 0 for d in deltas)
    assert deltas  # 至少回调一次


def test_n_workers_forwarded_results_equivalent() -> None:
    """n_workers=1（一次性单线程池）与 n_workers=None（全局池）逐候选一致。

    线程数不改变 evaluate_point 的纯函数结果；此测试证明 ThreadPoolBuilder
    路径工作且确定性不变（status / 整数索引 / 布尔字段精确相等）。
    """
    n_dep, n_alpha = 3, 4
    dep_states = _circular_orbit(0.9, 0.08, 40)[:n_dep]
    dep_times = np.linspace(0.0, 6.28, 40)[:n_dep]
    arrival_states = _circular_orbit(0.7, 0.12, 30)
    alpha_grid = np.linspace(0.9, 1.05, n_alpha)
    kw = _kwargs(dep_states, dep_times, alpha_grid, arrival_states)

    r_one = grid_search_rust(**kw, n_workers=1)
    r_def = grid_search_rust(**kw)  # n_workers=None → Rayon 全局池
    assert len(r_one) == len(r_def) == n_dep * n_alpha

    for i, (a, b) in enumerate(zip(r_one, r_def, strict=True)):
        assert a["status"] == b["status"], f"[{i}] status: {a['status']!r} vs {b['status']!r}"
        assert a["min_distance_idx"] == b["min_distance_idx"], f"[{i}] min_distance_idx"
        assert a["min_distance_orbit_idx"] == b["min_distance_orbit_idx"], (
            f"[{i}] min_distance_orbit_idx"
        )
        for f in ("success", "intersection_found", "collision_found", "local_minimum_found"):
            assert a[f] == b[f], f"[{i}] {f}: {a[f]!r} vs {b[f]!r}"


def test_trajectory_filtered_for_non_success() -> None:
    """非 success 候选不回传轨迹；success 候选轨迹非 None。

    远场景（arrival xc=50）全部 no_intersection，轨迹全 None；近场景（arrival
    与 dep 同心 xc=0.9、半径差 0.04<阈值）至少有一个 success 候选，其轨迹非 None。
    """
    n_dep, n_alpha = 3, 4
    dep_states = _circular_orbit(0.9, 0.08, 40)[:n_dep]
    dep_times = np.linspace(0.0, 6.28, 40)[:n_dep]
    alpha_grid = np.linspace(0.9, 1.05, n_alpha)

    # 远场景：arrival 远离 dep，全部 no_intersection → 轨迹全 None。
    arrival_far = _circular_orbit(50.0, 0.10, 30)
    r_far = grid_search_rust(**_kwargs(dep_states, dep_times, alpha_grid, arrival_far))
    assert len(r_far) == n_dep * n_alpha
    statuses_far = {r["status"] for r in r_far}
    assert "success" not in statuses_far, "远场景不应出现 success"
    for r in r_far:
        assert r["transfer_trajectory"] is None, (
            f"non-success 候选不应回传轨迹，status={r['status']!r}"
        )
        assert r["transfer_times"] is None

    # 近场景：arrival 与 dep 同心（xc=0.9），半径 0.12 vs dep 0.08，
    # 半径差 0.04 < min_distance_threshold=0.05，α=1.0 时轨迹最近距 ≈0.04 → success。
    arrival_near = _circular_orbit(0.9, 0.12, 30)
    r_near = grid_search_rust(**_kwargs(dep_states, dep_times, alpha_grid, arrival_near))
    assert len(r_near) == n_dep * n_alpha
    assert any(r["status"] == "success" for r in r_near), "近场景应至少有一个 success 候选"
    for r in r_near:
        if r["status"] == "success":
            assert r["transfer_trajectory"] is not None, "success 候选轨迹不应为 None"
            assert r["transfer_times"] is not None
        else:
            assert r["transfer_trajectory"] is None, (
                f"non-success 候选不应回传轨迹，status={r['status']!r}"
            )
            assert r["transfer_times"] is None
