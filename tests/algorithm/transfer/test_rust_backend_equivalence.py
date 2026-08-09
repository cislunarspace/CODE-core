"""Rust 串行网格搜索 vs Python sequential 等价性对照（阶段 B）。

对照权威基准 ``e2m2e.algorithm.transfer.search_parallel.dispatch_grid_search``
（``n_workers=1`` 走 ``grid_search_sequential``），逐候选验证 Rust 后端
``grid_search_rust_serial`` 的数值与索引约定一致。

- 整数索引字段（argmin / 首次命中步）：**精确相等**——分叉即算法不一致
  （非数值噪声）。Python sequential 与 Rust 都走同一个 Rust ``propagate_cr3bp``，
  states 逐位相同，故 argmin 必然一致。
- 布尔字段（success / intersection_found / ...）：精确相等
- 浮点字段（min_distance / dv_departure / ...）：``assert_allclose(rtol=1e-9, atol=1e-12)``

网格选得温和（窄 α 范围、短 transfer_time），不触发积分发散分支；若触发，
该候选 status=="integration_failed" 单独跳过 dv_departure 对照（Rust 走 1e10
惩罚，Python 保留真实 dv，设计文档 transfer-grid-search-rust.md:88 约定）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

# 扩展未构建时（doc build / 无 spice 构建合法）整模块跳过。
pytest.importorskip("e2m2e._integrators")

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.transfer import TransferSearch
from e2m2e.algorithm.transfer.search_parallel import (
    dispatch_grid_search,
    sample_departure_points,
)
from e2m2e.data.types.orbit import Orbit
from e2m2e.integrators import grid_search_rust, grid_search_rust_serial

pytestmark = pytest.mark.orchestration


MU = 1.21506683e-2  # 地月质量参数

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
    """构造绕 (xc, 0) 的平面圆轨道（CR3BP 无量纲），n 点等间隔采样。

    速度取逆时针圆周切向（与 test_dro_ro_search._simple_orbit 同范式），
    α 缩放切向速度后轨道仍稳定，不触发积分发散。
    """
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


def _sort_key(r: dict) -> tuple:
    """按 (departure_time, alpha) 排序；两边返回顺序本就一致，排序仅为保险。"""
    return (round(float(r["departure_time"]), 12), float(r["alpha"]))


def _assert_candidate_equal(py_r: dict, rs_r: dict) -> None:
    """逐字段对照单个候选解（Python sequential vs Rust serial）。"""
    # status 字符串精确相等（success/collision/no_intersection/integration_failed）。
    assert py_r["status"] == rs_r["status"], f"status: py={py_r['status']!r} rs={rs_r['status']!r}"
    # collision_body 字符串精确相等（None / "earth" / "moon"）。
    assert py_r["collision_body"] == rs_r["collision_body"], (
        f"collision_body: py={py_r['collision_body']!r} rs={rs_r['collision_body']!r}"
    )

    # 整数索引（含 None）：精确相等。分叉 = 算法不一致。
    for f in INT_FIELDS:
        assert py_r[f] == rs_r[f], f"{f}: py={py_r[f]!r} rs={rs_r[f]!r}"

    # 布尔：精确相等。
    for f in BOOL_FIELDS:
        assert py_r[f] == rs_r[f], f"{f}: py={py_r[f]!r} rs={rs_r[f]!r}"

    # 浮点：allclose（None / inf 特例）。
    integration_failed = py_r["status"] == "integration_failed"
    for f in FLOAT_FIELDS:
        py_v = py_r[f]
        rs_v = rs_r[f]
        if py_v is None:
            assert rs_v is None, f"{f}: py=None rs={rs_v!r}"
            continue
        assert rs_v is not None, f"{f}: py={py_v!r} rs=None"
        if np.isinf(py_v) or np.isnan(py_v):
            assert py_v == rs_v, f"{f}: py={py_v!r} rs={rs_v!r}"
            continue
        # 积分发散分支：Rust dv_departure=1e10 惩罚，Python 保留真实值，跳过。
        if integration_failed and f == "dv_departure":
            continue
        assert_allclose(rs_v, py_v, rtol=1e-9, atol=1e-12, err_msg=f"{f}")

    # departure_state：6 维向量逐位一致（同一出发点采样）。
    assert_allclose(
        rs_r["departure_state"],
        py_r["departure_state"],
        rtol=1e-12,
        atol=1e-12,
        err_msg="departure_state",
    )

    # transfer_trajectory：积分轨迹逐位一致（核心等价性证据——两边走同一
    # Rust propagate_cr3bp）。成功分支对照；失败分支两边都 None。
    py_traj = py_r.get("transfer_trajectory")
    rs_traj = rs_r.get("transfer_trajectory")
    if py_traj is None:
        assert rs_traj is None, "transfer_trajectory: py=None rs 非 None"
    else:
        assert rs_traj is not None, "transfer_trajectory: py 非 None rs=None"
        py_traj = np.asarray(py_traj, dtype=float)
        rs_traj = np.asarray(rs_traj, dtype=float)
        assert py_traj.shape == rs_traj.shape, (
            f"transfer_trajectory shape: py={py_traj.shape} rs={rs_traj.shape}"
        )
        assert_allclose(rs_traj, py_traj, rtol=1e-9, atol=1e-12, err_msg="transfer_trajectory")


def test_rust_serial_matches_python_sequential(
    searcher: TransferSearch,
    dep_orbit: Orbit,
    arr_orbit: Orbit,
) -> None:
    """同一个小网格分别跑 Rust 串行与 Python sequential，逐候选对照全过。"""
    # Python sequential（dispatch_grid_search 在 n_workers=1 时走 grid_search_sequential）。
    results_py = dispatch_grid_search(
        searcher,
        dep_orbit,
        arr_orbit,
        verbose=False,
        n_workers=1,
        parallel_backend="processes",
    )

    # Rust serial：出发点采样与 dispatch_grid_search 内部一致（同 departure_orbit）。
    dep_states, dep_times = sample_departure_points(dep_orbit, searcher.n_departure)
    alpha_grid = np.linspace(searcher.alpha_min, searcher.alpha_max, searcher.n_alpha)
    arrival_states = np.asarray(arr_orbit.states, dtype=float)
    results_rs = grid_search_rust_serial(
        dep_states,
        dep_times,
        alpha_grid,
        arrival_states,
        mu=float(searcher.mu),
        max_transfer_time=float(searcher.max_transfer_time),
        integration_dt=float(searcher.integration_dt),
        intersection_threshold=float(searcher.intersection_threshold),
        min_distance_threshold=float(searcher.min_distance_threshold),
        collision_earth_radius=float(searcher.collision_earth_radius),
        collision_moon_radius=float(searcher.collision_moon_radius),
        rtol=float(searcher.dynamics.rtol),
        atol=float(searcher.dynamics.atol),
        max_step=float(searcher.dynamics.max_step),
    )

    # 候选总数 = n_departure * n_alpha。
    n_expected = int(searcher.n_departure) * int(searcher.n_alpha)
    assert len(results_py) == n_expected
    assert len(results_rs) == n_expected

    # 排序后逐候选对照（两边返回顺序本就一致：外 departure、内 alpha）。
    results_py.sort(key=_sort_key)
    results_rs.sort(key=_sort_key)
    for py_r, rs_r in zip(results_py, results_rs, strict=True):
        _assert_candidate_equal(py_r, rs_r)


def test_rust_serial_preserves_grid_order(
    searcher: TransferSearch,
    dep_orbit: Orbit,
    arr_orbit: Orbit,
) -> None:
    """Rust 串行返回顺序：外层 departure、内层 alpha（与 grid_search_sequential 一致）。

    钉死保序不变量——阶段 C 加 Rayon par_iter 时，collect 须保持此顺序
    （E2M2E_SEARCH_PARALLEL=0 串行模式对照基准）。
    """
    dep_states, dep_times = sample_departure_points(dep_orbit, searcher.n_departure)
    alpha_grid = np.linspace(searcher.alpha_min, searcher.alpha_max, searcher.n_alpha)
    arrival_states = np.asarray(arr_orbit.states, dtype=float)
    results_rs = grid_search_rust_serial(
        dep_states,
        dep_times,
        alpha_grid,
        arrival_states,
        mu=float(searcher.mu),
        max_transfer_time=float(searcher.max_transfer_time),
        integration_dt=float(searcher.integration_dt),
        intersection_threshold=float(searcher.intersection_threshold),
        min_distance_threshold=float(searcher.min_distance_threshold),
        collision_earth_radius=float(searcher.collision_earth_radius),
        collision_moon_radius=float(searcher.collision_moon_radius),
        rtol=float(searcher.dynamics.rtol),
        atol=float(searcher.dynamics.atol),
        max_step=float(searcher.dynamics.max_step),
    )

    n_dep = int(searcher.n_departure)
    n_alpha = int(searcher.n_alpha)
    assert len(results_rs) == n_dep * n_alpha
    for idx, r in enumerate(results_rs):
        i_dep = idx // n_alpha
        i_alpha = idx % n_alpha
        # departure_time 与采样点一致；alpha 与网格点一致。
        assert r["departure_time"] == pytest.approx(dep_times[i_dep])
        assert r["alpha"] == pytest.approx(alpha_grid[i_alpha])


def _run_grid_search(
    searcher: TransferSearch,
    dep_orbit: Orbit,
    arr_orbit: Orbit,
    parallel: bool | None,
) -> list[dict[str, Any]]:
    """展平网格输入 → 调 grid_search_rust（阶段 C 并行入口）。"""
    dep_states, dep_times = sample_departure_points(dep_orbit, searcher.n_departure)
    alpha_grid = np.linspace(searcher.alpha_min, searcher.alpha_max, searcher.n_alpha)
    arrival_states = np.asarray(arr_orbit.states, dtype=float)
    return grid_search_rust(
        dep_states,
        dep_times,
        alpha_grid,
        arrival_states,
        mu=float(searcher.mu),
        max_transfer_time=float(searcher.max_transfer_time),
        integration_dt=float(searcher.integration_dt),
        intersection_threshold=float(searcher.intersection_threshold),
        min_distance_threshold=float(searcher.min_distance_threshold),
        collision_earth_radius=float(searcher.collision_earth_radius),
        collision_moon_radius=float(searcher.collision_moon_radius),
        rtol=float(searcher.dynamics.rtol),
        atol=float(searcher.dynamics.atol),
        max_step=float(searcher.dynamics.max_step),
        parallel=parallel,
    )


def test_parallel_equals_serial(
    searcher: TransferSearch,
    dep_orbit: Orbit,
    arr_orbit: Orbit,
) -> None:
    """阶段 C：并行（Rayon par_iter）与串行逐候选逐字段位级一致。

    par_iter + collect 保序，evaluate_point 是纯函数（直接调纯 Rust
    ``propagate_cr3bp``，CR3BP 纯数学无 SPICE FFI），同一输入下两边浮点
    运算完全相同——故可断言轨迹逐位相等、标量精确相等。
    """
    results_serial = _run_grid_search(searcher, dep_orbit, arr_orbit, parallel=False)
    results_parallel = _run_grid_search(searcher, dep_orbit, arr_orbit, parallel=True)

    n_expected = int(searcher.n_departure) * int(searcher.n_alpha)
    assert len(results_serial) == n_expected
    assert len(results_parallel) == n_expected

    for idx, (s, p) in enumerate(zip(results_serial, results_parallel, strict=True)):
        # 保序：顺序一致（无需排序），逐候选对照。
        # 整数 / 布尔 / 字符串字段：精确相等。
        assert s["status"] == p["status"], f"[{idx}] status: s={s['status']!r} p={p['status']!r}"
        assert s["collision_body"] == p["collision_body"], f"[{idx}] collision_body"
        for f in INT_FIELDS:
            assert s[f] == p[f], f"[{idx}] {f}: s={s[f]!r} p={p[f]!r}"
        for f in BOOL_FIELDS:
            assert s[f] == p[f], f"[{idx}] {f}: s={s[f]!r} p={p[f]!r}"
        # 浮点标量：位级相等（含 None / inf 特例）。
        for f in FLOAT_FIELDS:
            sv, pv = s[f], p[f]
            if sv is None:
                assert pv is None, f"[{idx}] {f}: s=None p={pv!r}"
                continue
            assert pv is not None, f"[{idx}] {f}: s={sv!r} p=None"
            if np.isinf(sv) or np.isnan(sv):
                assert sv == pv, f"[{idx}] {f}: s={sv!r} p={pv!r}"
                continue
            assert float(sv) == float(pv), f"[{idx}] {f}: s={sv!r} p={pv!r}"
        # departure_state：6 维逐位相等。
        np.testing.assert_array_equal(
            s["departure_state"], p["departure_state"], err_msg=f"[{idx}] departure_state"
        )
        # 轨迹：逐位相等（核心等价性证据——同走 propagate_cr3bp）。
        st, pt = s.get("transfer_trajectory"), p.get("transfer_trajectory")
        if st is None:
            assert pt is None, f"[{idx}] transfer_trajectory: s=None p 非 None"
        else:
            assert pt is not None, f"[{idx}] transfer_trajectory: s 非 None p=None"
            st = np.asarray(st, dtype=float)
            pt = np.asarray(pt, dtype=float)
            assert st.shape == pt.shape, f"[{idx}] traj shape: s={st.shape} p={pt.shape}"
            np.testing.assert_array_equal(st, pt, err_msg=f"[{idx}] transfer_trajectory")
