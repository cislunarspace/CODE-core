"""轨道搜索的并行/串行后端与 per-α 积分内核。

per-α 积分内核 :func:`search_single_departure` 通过 ``searcher`` 上的几何方法
分发（``searcher._forward_integrate`` / ``_check_collision`` / ``_compute_distance_series`` 等），
便于 ``monkeypatch.setattr(TransferSearch, "_x", ...)`` 注入合成行为。
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from ...data.types.orbit import Orbit
from ..dynamics import CR3BP_Dynamics, CR3BP_System
from .search_progress import (
    AggregatePbarWithSlot,
    open_parallel_worker_progress_bars,
    open_search_progress_bar,
    reset_tqdm_bar,
    use_multiline_worker_tqdm,
)

if TYPE_CHECKING:
    from .transfer_search import TransferSearch

logger = logging.getLogger(__name__)


def sample_departure_points(
    departure_orbit: Orbit, n_departure: int
) -> tuple[np.ndarray, np.ndarray]:
    """从轨道星历等时间间隔下采样出发点。"""
    times = departure_orbit.times
    states = departure_orbit.states
    n_pts = len(times)
    if n_pts == 0:
        raise ValueError("出发轨道无数据点")
    if n_departure is None:
        raise ValueError("n_departure 未设置")
    n = int(n_departure)
    if n <= 0:
        raise ValueError("n_departure 须为正整数")
    if n > n_pts:
        raise ValueError(
            f"n_departure（{n}）不能大于出发轨道星历点数（{n_pts}），"
            f"请减小 n_departure 或增加星历密度"
        )
    if n == 1:
        idx = np.array([0], dtype=int)
    else:
        idx = (np.arange(n, dtype=float) * (n_pts - 1) / (n - 1)).round().astype(int)
    return states[idx].copy(), times[idx].copy()


def forward_integrate(
    dynamics: CR3BP_Dynamics,
    initial_state: np.ndarray,
    transfer_time: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """CR3BP 下从 ``initial_state`` 前向积分到 ``transfer_time``，得到等间隔采样轨迹。"""
    n_steps = max(int(transfer_time / dt) + 1, 2)
    t_eval = np.linspace(0, transfer_time, n_steps)
    result = dynamics.propagate(
        initial_state=initial_state,
        t_span=(0.0, transfer_time),
        t_eval=t_eval,
        with_stm=False,
        with_jacobi=False,
    )
    return result["states"], result["time"]


def search_single_departure(
    searcher: TransferSearch,
    *,
    departure_state: np.ndarray,
    departure_time: float,
    arrival_orbit: Orbit,
    verbose: bool = False,
    pbar: Any | None = None,
    departure_index: int | None = None,
    progress_queue: Any | None = None,
) -> list[dict[str, Any]]:
    """对单出发点执行 α 网格搜索。"""
    a0 = searcher.alpha_min
    a1 = searcher.alpha_max
    na = searcher.n_alpha
    mtt = searcher.max_transfer_time
    idt = searcher.integration_dt
    ith = searcher.intersection_threshold
    mdt = searcher.min_distance_threshold
    if (
        a0 is None
        or a1 is None
        or na is None
        or mtt is None
        or idt is None
        or ith is None
        or mdt is None
    ):
        raise ValueError(
            "请先设置 alpha_min, alpha_max, n_alpha, max_transfer_time, integration_dt, "
            "intersection_threshold, min_distance_threshold"
        )
    alpha_grid = np.linspace(a0, a1, na)
    n_alpha = int(na)

    results: list[dict[str, Any]] = []
    for i_alpha, alpha in enumerate(alpha_grid, start=1):
        try:
            if verbose:
                pct = i_alpha / n_alpha * 100
                print(f"    α 进度: {i_alpha}/{n_alpha} ({pct:.1f}%)", flush=True)
            new_vel = searcher._compute_departure_velocity(departure_state, alpha)
            dv_departure = float(np.linalg.norm(new_vel - departure_state[3:6]))
            initial_state = np.concatenate([departure_state[:3], new_vel])

            try:
                traj_states, traj_times = searcher._forward_integrate(initial_state, mtt, idt)
            except Exception:
                logger.debug(
                    "积分失败: alpha=%.4f, departure_time=%.4f",
                    alpha,
                    departure_time,
                    exc_info=True,
                )
                results.append(
                    {
                        "success": False,
                        "departure_state": departure_state,
                        "departure_time": departure_time,
                        "alpha": alpha,
                        "status": "integration_failed",
                        "dv_departure": dv_departure,
                        "dv_insertion": None,
                        "min_distance_orbit_idx": None,
                        "first_intersection_idx": None,
                        "first_intersection_time": None,
                        "first_min_distance_idx": None,
                        "first_min_distance_time": None,
                    }
                )
                continue

            collision, body, col_idx = searcher._check_collision(traj_states)
            d_per_step, orbit_idx_per_step = searcher._compute_distance_series(
                traj_states, arrival_orbit
            )
            min_idx = int(np.argmin(d_per_step))
            min_dist = float(d_per_step[min_idx])
            orbit_idx = int(orbit_idx_per_step[min_idx])
            v_tr = traj_states[min_idx][3:6]
            v_ro = arrival_orbit.states[orbit_idx][3:6]
            dv_insertion = float(np.linalg.norm(v_tr - v_ro))
            intersection, int_point, int_idx = searcher._detect_intersection(
                traj_states, arrival_orbit, ith
            )
            local_min, local_min_dist, local_min_idx = searcher._detect_local_minimum(
                traj_states, arrival_orbit
            )

            first_int_idx: int | None
            first_int_time: float | None
            first_md_idx: int | None
            first_md_time: float | None
            _int_hits = np.where(d_per_step <= ith)[0]
            if _int_hits.size > 0:
                first_int_idx = int(_int_hits[0])
                first_int_time = float(traj_times[first_int_idx])
            else:
                first_int_idx = None
                first_int_time = None
            _md_hits = np.where(d_per_step <= mdt)[0]
            if _md_hits.size > 0:
                first_md_idx = int(_md_hits[0])
                first_md_time = float(traj_times[first_md_idx])
            else:
                first_md_idx = None
                first_md_time = None

            result = {
                "success": True,
                "departure_state": departure_state,
                "departure_time": departure_time,
                "alpha": alpha,
                "transfer_trajectory": traj_states,
                "transfer_times": traj_times,
                "transfer_time": traj_times[-1],
                "min_distance": min_dist,
                "min_distance_idx": min_idx,
                "min_distance_orbit_idx": int(orbit_idx),
                "dv_departure": dv_departure,
                "dv_insertion": dv_insertion,
                "intersection_found": intersection,
                "intersection_point": int_point,
                "intersection_idx": int_idx,
                "first_intersection_idx": first_int_idx,
                "first_intersection_time": first_int_time,
                "first_min_distance_idx": first_md_idx,
                "first_min_distance_time": first_md_time,
                "local_minimum_found": local_min,
                "local_minimum_distance": local_min_dist,
                "local_minimum_idx": local_min_idx,
                "collision_found": collision,
                "collision_body": body,
                "collision_idx": col_idx,
            }
            if collision:
                result["status"] = "collision"
            elif intersection or min_dist < mdt:
                result["status"] = "success"
            else:
                result["status"] = "no_intersection"
            results.append(result)
        finally:
            if pbar is not None:
                pbar.update(1)
                if departure_index is not None:
                    pbar.set_postfix_str(f"dep={departure_index} α={alpha:.4f}")
            if progress_queue is not None:
                progress_queue.put(1)
    return results


def dispatch_grid_search(
    searcher: TransferSearch,
    departure_orbit: Orbit,
    arrival_orbit: Orbit,
    verbose: bool,
    n_workers: int | None,
    parallel_backend: str,
) -> list[dict[str, Any]]:
    """网格搜索分发。"""
    dep_name = getattr(departure_orbit, "name", "unknown")
    arr_name = getattr(arrival_orbit, "name", "unknown")
    departure_states, departure_times = sample_departure_points(
        departure_orbit, searcher.n_departure
    )
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    pb = parallel_backend.strip().lower()
    if pb not in ("processes", "threads", "rust"):
        raise ValueError("parallel_backend 须为 'processes'、'threads' 或 'rust'")
    if pb == "rust":
        return grid_search_rust_dispatch(
            searcher,
            departure_orbit,
            arrival_orbit,
            departure_states,
            departure_times,
            dep_name,
            arr_name,
            verbose,
            n_workers,
        )
    if n_workers == 1:
        return grid_search_sequential(
            searcher,
            departure_states,
            departure_times,
            arrival_orbit,
            dep_name,
            arr_name,
            verbose,
        )
    if pb == "processes":
        return grid_search_parallel_processes(
            searcher,
            departure_states,
            departure_times,
            arrival_orbit,
            dep_name,
            arr_name,
            verbose,
            n_workers,
        )
    return grid_search_parallel_threads(
        searcher,
        departure_states,
        departure_times,
        arrival_orbit,
        dep_name,
        arr_name,
        verbose,
        n_workers,
    )


def _geometry_methods_monkeypatched(searcher: TransferSearch) -> bool:
    """检测 ``TransferSearch`` 的三个几何方法是否被 ``monkeypatch.setattr`` 替换。

    Rust 后端整体下沉积分/碰撞/局部极小三步，不经过 Python 方法分发；若这些方法
    被 patch（测试注入合成行为），直接走 Rust 会让 patch 形同虚设。判定依据：原始
    方法定义在 ``transfer_search`` 模块，``__qualname__`` 形如
    ``TransferSearch._forward_integrate``；测试注入的函数定义在别处，qualname 必然不同。
    """
    for name in ("_forward_integrate", "_check_collision", "_detect_local_minimum"):
        method = getattr(type(searcher), name, None)
        if method is None:
            continue
        if getattr(method, "__qualname__", None) != f"TransferSearch.{name}":
            return True
    return False


def grid_search_rust_dispatch(
    searcher: TransferSearch,
    departure_orbit: Orbit,
    arrival_orbit: Orbit,
    departure_states: np.ndarray,
    departure_times: np.ndarray,
    dep_name: str,
    arr_name: str,
    verbose: bool,
    n_workers: int,
) -> list[dict[str, Any]]:
    """Rust + Rayon 后端网格搜索分发（阶段 D1）。

    展平 POD 输入喂给 :func:`e2m2e.integrators.grid_search_rust`（``py.allow_threads``
    释放 GIL + Rayon ``par_iter``），拿回候选解后追加 ``departure_time_index`` /
    ``departure_orbit_name`` / ``arrival_orbit_name``，对齐 Python sequential 后端字段。

    两种情况回退 Python 路径（结果正确，仅降速）：

    - **monkeypatch 缝**：几何方法被替换时（见 :func:`_geometry_methods_monkeypatched`），
      Rust 内核不经过 Python 分发，patch 不生效——回退保住测试语义。
    - **Rust 扩展缺失**：``grid_search_rust`` 抛 ``RuntimeError``（``transfer_grid_search_py``
      为 None），回退 ``processes`` 后端。

    Rust 总是走 Rayon 多核并行（``parallel=True``）——网格搜索的目标是快速完成，
    Rayon 进程内线程池默认用满 cpu_count 个线程。要限制线程数用 ``RAYON_NUM_THREADS``
    环境变量。``n_workers`` 仅用于 monkeypatch 回退时的 Python 后端选择。
    """
    if _geometry_methods_monkeypatched(searcher):
        backend = "sequential" if n_workers == 1 else "processes"
        if verbose:
            logger.info("Rust 后端检测到几何方法被 monkeypatch，回退 Python（%s）", backend)
        if n_workers == 1:
            return grid_search_sequential(
                searcher,
                departure_states,
                departure_times,
                arrival_orbit,
                dep_name,
                arr_name,
                verbose,
            )
        return grid_search_parallel_processes(
            searcher,
            departure_states,
            departure_times,
            arrival_orbit,
            dep_name,
            arr_name,
            verbose,
            n_workers,
        )

    if (
        searcher.alpha_min is None
        or searcher.alpha_max is None
        or searcher.n_alpha is None
        or searcher.max_transfer_time is None
        or searcher.integration_dt is None
        or searcher.intersection_threshold is None
        or searcher.min_distance_threshold is None
        or searcher.collision_earth_radius is None
        or searcher.collision_moon_radius is None
    ):
        raise ValueError(
            "请先设置 alpha_min, alpha_max, n_alpha, max_transfer_time, integration_dt, "
            "intersection_threshold, min_distance_threshold, collision_earth_radius, "
            "collision_moon_radius"
        )
    alpha_grid = np.linspace(searcher.alpha_min, searcher.alpha_max, int(searcher.n_alpha))
    arrival_states = np.asarray(arrival_orbit.states, dtype=float)
    n_alpha = int(searcher.n_alpha)
    parallel = True  # Rust Rayon 进程内线程池总是多核并行（网格搜索目标：快速完成）

    try:
        from ...integrators import grid_search_rust

        results = grid_search_rust(
            departure_states,
            departure_times,
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
    except (ImportError, RuntimeError) as exc:
        if verbose:
            logger.info("Rust 后端不可用（%s），回退 processes", exc)
        return grid_search_parallel_processes(
            searcher,
            departure_states,
            departure_times,
            arrival_orbit,
            dep_name,
            arr_name,
            verbose,
            n_workers,
        )

    for idx, r in enumerate(results):
        r["departure_time_index"] = idx // n_alpha
        r["departure_orbit_name"] = dep_name
        r["arrival_orbit_name"] = arr_name

    if verbose:
        mode = "Rayon 并行" if parallel else "Rust 串行"
        print(
            f"  Rust 后端（{mode}）: {len(departure_states)}×{n_alpha}={len(results)} 评估",
            flush=True,
        )
    return results


def grid_search_sequential(
    searcher: TransferSearch,
    departure_states: np.ndarray,
    departure_times: np.ndarray,
    arrival_orbit: Orbit,
    dep_name: str,
    arr_name: str,
    verbose: bool,
) -> list[dict[str, Any]]:
    """串行网格搜索。"""
    all_results: list[dict[str, Any]] = []
    total_departures = len(departure_states)
    n_alpha_v = searcher.n_alpha
    if n_alpha_v is None:
        raise ValueError("n_alpha 未设置")
    n_alpha = int(n_alpha_v)
    total_steps = total_departures * n_alpha
    pbar = None
    if verbose and total_steps > 0:
        pbar = open_search_progress_bar(total_steps, "网格搜索")
    if verbose and total_steps <= 0:
        print(f"  总迭代步数: {total_steps}（出发点 × α），无进度条", flush=True)

    try:
        for i, (dep_state, dep_time) in enumerate(
            zip(departure_states, departure_times, strict=False)
        ):
            results = searcher._search_single_departure(
                departure_state=dep_state,
                departure_time=dep_time,
                arrival_orbit=arrival_orbit,
                verbose=(verbose and pbar is None),
                pbar=pbar,
                departure_index=i,
            )
            for r in results:
                r["departure_orbit_name"] = dep_name
                r["arrival_orbit_name"] = arr_name
                r["departure_time_index"] = i
            all_results.extend(results)
    finally:
        if pbar is not None:
            pbar.close()
    return all_results


def _process_pack_base(searcher):
    """从 searcher 读取多进程 worker 打包配置。"""
    dyn = searcher.dynamics
    return (
        searcher.mu,
        float(searcher.alpha_min),
        float(searcher.alpha_max),
        int(searcher.n_alpha),
        int(searcher.n_departure),
        float(searcher.max_transfer_time),
        float(searcher.intersection_threshold),
        float(searcher.min_distance_threshold),
        float(searcher.collision_earth_radius),
        float(searcher.collision_moon_radius),
        float(searcher.integration_dt),
        str(dyn.integrator),
        float(dyn.rtol),
        float(dyn.atol),
        float(dyn.max_step),
    )


def grid_search_parallel_processes(
    searcher: TransferSearch,
    departure_states: np.ndarray,
    departure_times: np.ndarray,
    arrival_orbit: Orbit,
    dep_name: str,
    arr_name: str,
    verbose: bool,
    n_workers: int,
) -> list[dict[str, Any]]:
    """多进程并行：每进程独立 Python 解释器，利于 CPU 密集积分。"""
    total_departures = len(departure_states)
    n_alpha_v = searcher.n_alpha
    if n_alpha_v is None:
        raise ValueError("n_alpha 未设置")
    n_alpha = int(n_alpha_v)
    total_steps = total_departures * n_alpha
    all_results: list[dict[str, Any]] = []

    arrival_states = np.asarray(arrival_orbit.states)
    arrival_times_a = np.asarray(arrival_orbit.times)
    ap = getattr(arrival_orbit, "period", None)
    arrival_period: float | None = float(ap) if ap is not None else None

    pbar = None
    progress_queue: Any | None = None
    progress_manager: Any | None = None
    poll_stop: threading.Event | None = None
    poll_thread: threading.Thread | None = None

    if verbose and total_steps > 0:
        tqdm.write(
            f"  并行搜索(进程): {total_departures}×{n_alpha}={total_steps} 步 | {n_workers} 进程"
        )
        pbar = open_search_progress_bar(total_steps, "并行网格搜索(进程)")
        # 多进程下 tqdm 无法跨进程；用 Manager().Queue() 中转。
        progress_manager = multiprocessing.Manager()
        progress_queue = progress_manager.Queue()
        poll_stop = threading.Event()

        def _poll() -> None:
            assert pbar is not None
            while True:
                if poll_stop.is_set():
                    break
                try:
                    progress_queue.get(timeout=0.25)
                    pbar.update(1)
                except queue.Empty:
                    continue
            while True:
                try:
                    progress_queue.get_nowait()
                    pbar.update(1)
                except queue.Empty:
                    break

        poll_thread = threading.Thread(target=_poll, daemon=True)
        poll_thread.start()

    pack_base = _process_pack_base(searcher) + (dep_name, arr_name)
    if (
        searcher.alpha_min is None
        or searcher.alpha_max is None
        or searcher.n_alpha is None
        or searcher.n_departure is None
        or searcher.max_transfer_time is None
        or searcher.intersection_threshold is None
        or searcher.min_distance_threshold is None
        or searcher.collision_earth_radius is None
        or searcher.collision_moon_radius is None
        or searcher.integration_dt is None
    ):
        raise ValueError(
            "请先设置 alpha_min, alpha_max, n_alpha, n_departure, max_transfer_time, "
            "intersection_threshold, min_distance_threshold, collision_earth_radius, "
            "collision_moon_radius, integration_dt"
        )

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for i, (dep_state, dep_time) in enumerate(
                zip(departure_states, departure_times, strict=False)
            ):
                packed = (
                    (
                        i,
                        np.asarray(dep_state, dtype=float),
                        float(dep_time),
                        arrival_states,
                        arrival_times_a,
                        arrival_period,
                    )
                    + pack_base
                    + (progress_queue,)
                )
                fut = executor.submit(process_departure_worker_packed, packed)
                futures[fut] = i
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results = fut.result()
                    all_results.extend(results)
                except Exception as e:
                    if verbose:
                        tqdm.write(f"    出发点 {idx} 处理失败: {e}")
    finally:
        if poll_stop is not None:
            poll_stop.set()
        if poll_thread is not None:
            poll_thread.join(timeout=30.0)
        if pbar is not None and progress_queue is not None:
            while True:
                try:
                    progress_queue.get_nowait()
                    pbar.update(1)
                except queue.Empty:
                    break
        if pbar is not None:
            pbar.close()
        if progress_manager is not None:
            progress_manager.shutdown()
    return all_results


def _run_departure_with_worker_slot(
    searcher: TransferSearch,
    slot_queue: queue.Queue[int],
    n_alpha: int,
    departure_index: int,
    departure_state: np.ndarray,
    departure_time: float,
    arrival_orbit: Orbit,
    worker_bars: list[Any] | None,
    aggregate_pbar: Any | None,
    aggregate_lock: threading.Lock | None,
) -> list[dict[str, Any]]:
    """取槽 → 跑单出发点 α 网格 → 还槽。"""
    slot = slot_queue.get()
    try:
        if worker_bars is not None:
            bar = worker_bars[slot]
            reset_tqdm_bar(bar, n_alpha)
            bar.set_description_str(f"W{slot} dep={departure_index}", refresh=False)
            pbar: Any = bar
        elif aggregate_pbar is not None:
            pbar = AggregatePbarWithSlot(aggregate_pbar, aggregate_lock, slot)
        else:
            raise RuntimeError("worker_bars 与 aggregate_pbar 至少传入其一")
        return searcher._search_single_departure(
            departure_state=departure_state,
            departure_time=departure_time,
            arrival_orbit=arrival_orbit,
            verbose=False,
            pbar=pbar,
            departure_index=departure_index,
        )
    finally:
        slot_queue.put(slot)


def grid_search_parallel_threads(
    searcher: TransferSearch,
    departure_states: np.ndarray,
    departure_times: np.ndarray,
    arrival_orbit: Orbit,
    dep_name: str,
    arr_name: str,
    verbose: bool,
    n_workers: int,
) -> list[dict[str, Any]]:
    """多线程并行搜索：支持细粒度 tqdm 进度条。"""
    total_departures = len(departure_states)
    n_alpha_v = searcher.n_alpha
    if n_alpha_v is None:
        raise ValueError("n_alpha 未设置")
    n_alpha = int(n_alpha_v)
    total_steps = total_departures * n_alpha
    all_results: list[dict[str, Any]] = []

    worker_bars: list[Any] | None = None
    aggregate_pbar: Any | None = None
    aggregate_lock: threading.Lock | None = None
    slot_queue: queue.Queue[int] | None = None

    if verbose and total_steps > 0:
        use_multiline = use_multiline_worker_tqdm(n_workers)
        if use_multiline:
            tqdm.write(
                f"  并行搜索: {total_departures}×{n_alpha}={total_steps} 步"
                f" | {n_workers} 槽（分槽）"
            )
            worker_bars = open_parallel_worker_progress_bars(n_workers, n_alpha)
        else:
            tqdm.write(
                f"  并行搜索: {total_departures}×{n_alpha}={total_steps} 步"
                f" | {n_workers} 槽（单行）"
            )
            aggregate_lock = threading.Lock()
            aggregate_pbar = open_search_progress_bar(total_steps, "并行网格搜索")
        slot_queue = queue.Queue()
        for slot in range(n_workers):
            slot_queue.put(slot)

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            if worker_bars is not None and slot_queue is not None:
                futures = {
                    executor.submit(
                        _run_departure_with_worker_slot,
                        searcher,
                        slot_queue,
                        n_alpha,
                        i,
                        dep_state,
                        dep_time,
                        arrival_orbit,
                        worker_bars,
                        None,
                        None,
                    ): i
                    for i, (dep_state, dep_time) in enumerate(
                        zip(departure_states, departure_times, strict=False)
                    )
                }
            elif aggregate_pbar is not None and slot_queue is not None:
                futures = {
                    executor.submit(
                        _run_departure_with_worker_slot,
                        searcher,
                        slot_queue,
                        n_alpha,
                        i,
                        dep_state,
                        dep_time,
                        arrival_orbit,
                        None,
                        aggregate_pbar,
                        aggregate_lock,
                    ): i
                    for i, (dep_state, dep_time) in enumerate(
                        zip(departure_states, departure_times, strict=False)
                    )
                }
            else:
                futures = {
                    executor.submit(
                        searcher._search_single_departure,
                        dep_state,
                        dep_time,
                        arrival_orbit,
                        False,
                        None,
                        i,
                    ): i
                    for i, (dep_state, dep_time) in enumerate(
                        zip(departure_states, departure_times, strict=False)
                    )
                }
            for future in as_completed(futures):
                try:
                    results = future.result()
                    for r in results:
                        r["departure_orbit_name"] = dep_name
                        r["arrival_orbit_name"] = arr_name
                        r["departure_time_index"] = futures[future]
                    all_results.extend(results)
                except Exception as e:
                    if verbose:
                        tqdm.write(f"    出发点 {futures[future]} 处理失败: {e}")
    finally:
        if worker_bars is not None:
            for bar in worker_bars:
                bar.close()
        if aggregate_pbar is not None:
            aggregate_pbar.close()
    return all_results


def process_departure_worker(
    idx: int,
    dep_state: np.ndarray,
    dep_time: float,
    arrival_states: np.ndarray,
    arrival_times: np.ndarray,
    arrival_period: float | None,
    mu: float,
    alpha_min: float,
    alpha_max: float,
    n_alpha: int,
    n_departure: int,
    max_transfer_time: float,
    intersection_threshold: float,
    min_distance_threshold: float,
    collision_earth_radius: float,
    collision_moon_radius: float,
    integration_dt: float,
    integrator: str,
    rtol: float,
    atol: float,
    max_step: float,
    dep_name: str,
    arr_name: str,
    progress_queue: Any | None = None,
) -> list[dict[str, Any]]:
    """子进程入口（模块级，便于 Windows spawn 下 pickle）。"""
    from .transfer_search import TransferSearch

    arrival_orbit = Orbit(states=arrival_states, times=arrival_times)
    if arrival_period is not None:
        arrival_orbit.period = float(arrival_period)

    system = CR3BP_System(mu=mu, primary="earth", secondary="moon")
    dynamics = CR3BP_Dynamics(system=system)
    dynamics.integrator = integrator
    dynamics.rtol = rtol
    dynamics.atol = atol
    dynamics.max_step = max_step

    searcher = TransferSearch(dynamics=dynamics)
    searcher.configure_search(
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        n_alpha=n_alpha,
        n_departure=n_departure,
        max_transfer_time=max_transfer_time,
        intersection_threshold=intersection_threshold,
        min_distance_threshold=min_distance_threshold,
        collision_earth_radius=collision_earth_radius,
        collision_moon_radius=collision_moon_radius,
        integration_dt=integration_dt,
    )

    results = searcher._search_single_departure(
        departure_state=dep_state,
        departure_time=dep_time,
        arrival_orbit=arrival_orbit,
        progress_queue=progress_queue,
    )
    for r in results:
        r["departure_orbit_name"] = dep_name
        r["arrival_orbit_name"] = arr_name
        r["departure_time_index"] = idx
    return results


def process_departure_worker_packed(packed: tuple[Any, ...]) -> list[dict[str, Any]]:
    """单元组打包，供 ``ProcessPoolExecutor`` 提交。"""
    return process_departure_worker(*packed)
