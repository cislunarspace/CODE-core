"""
DRO → GEO 球面转移搜索

继承 TransferSearch，将到达条件从"与 RO 轨道相交/接近"替换为
"穿越 GEO 球面（r_GEO ≈ 0.10968 DU）"。

使用方式::

    from e2m2e.transfer import GeoTransferSearch

    searcher = GeoTransferSearch(dynamics)
    results = searcher.search(
        departure_orbit=dro_orbit,
        alpha_min=0.5, alpha_max=2.5, n_alpha=100,
        n_departure=200, max_transfer_time=100.0 / TU,
        geo_threshold=100.0 / DU,
        n_workers=None,
    )
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from e2m2e.transfer.transfer_search import TransferSearch, _process_departure_worker_packed

# ── GEO 常量 ──
DU = 3.84405e5
R_GEO_KM = 42164.0
R_GEO = R_GEO_KM / DU
MU_DEFAULT = 1.21506683e-2
V_CIRCULAR_GEO = np.sqrt((1.0 - MU_DEFAULT) / R_GEO)


# ── GEO 辅助函数 ──


def _geo_circular_velocity(pos: np.ndarray, mu: float) -> np.ndarray:
    """旋转系下 GEO 圆轨道速度。"""
    earth_center = np.array([-mu, 0.0, 0.0])
    r_rel = pos - earth_center
    r_xy = np.sqrt(r_rel[0] ** 2 + r_rel[1] ** 2)
    if r_xy < 1e-12:
        return np.array([0.0, V_CIRCULAR_GEO, 0.0])
    tangential = np.array([-r_rel[1], r_rel[0], 0.0]) / r_xy
    v_inertial = V_CIRCULAR_GEO * tangential
    omega_cross_r = np.array([-pos[1], pos[0], 0.0])
    return v_inertial - omega_cross_r


def _detect_sphere_crossing(
    states: np.ndarray,
    r_sphere: float,
    center: np.ndarray,
) -> Tuple[bool, int]:
    """检测轨迹首次穿越球面。"""
    dists = np.linalg.norm(states[:, :3] - center, axis=1)
    sign_changes = np.where(np.diff(np.sign(dists - r_sphere)))[0]
    if len(sign_changes) > 0:
        return True, int(sign_changes[0])
    return False, -1


def _closest_approach_to_sphere(
    states: np.ndarray,
    r_sphere: float,
    center: np.ndarray,
) -> Tuple[float, int]:
    """轨迹到球面的最近距离。"""
    dists = np.linalg.norm(states[:, :3] - center, axis=1)
    sphere_dists = np.abs(dists - r_sphere)
    idx = int(np.argmin(sphere_dists))
    return float(sphere_dists[idx]), idx


# ── GEO 并行 worker（模块级，可 pickle） ──


def _geo_search_worker(packed: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    """子进程入口，结构与 _process_departure_worker 一致。"""
    from e2m2e.core.system import CR3BP_System
    from e2m2e.core.dynamics import CR3BP_Dynamics

    (
        dep_idx,
        dep_state,
        dep_time,
        mu,
        geo_threshold,
        alpha_min,
        alpha_max,
        n_alpha,
        n_departure,
        max_transfer_time,
        intersection_threshold,
        min_distance_threshold,
        collision_earth_radius,
        collision_moon_radius,
        integration_dt,
        integrator,
        rtol,
        atol,
        max_step,
        dep_name,
        arr_name,
        progress_queue,
    ) = packed

    system = CR3BP_System(mu=mu, primary="earth", secondary="moon")
    dynamics = CR3BP_Dynamics(system=system)
    dynamics.integrator = integrator
    dynamics.rtol = rtol
    dynamics.atol = atol
    dynamics.max_step = max_step

    earth_center = np.array([-mu, 0.0, 0.0])
    r_geo = R_GEO

    alpha_grid = np.linspace(alpha_min, alpha_max, n_alpha)
    results: List[Dict[str, Any]] = []

    for alpha in alpha_grid:
        new_vel = TransferSearch._compute_departure_velocity(None, dep_state, alpha)
        dv_dep = float(np.linalg.norm(new_vel - dep_state[3:6]))
        x0 = np.concatenate([dep_state[:3], new_vel])

        try:
            n_steps = max(int(max_transfer_time / integration_dt) + 1, 2)
            t_eval = np.linspace(0.0, max_transfer_time, n_steps)
            prop = dynamics.propagate(
                initial_state=x0,
                t_span=(0.0, max_transfer_time),
                t_eval=t_eval,
                with_stm=False,
                with_jacobi=False,
            )
            states, times = prop["states"], prop["time"]
        except Exception:
            results.append(
                {
                    "success": False,
                    "departure_state": dep_state,
                    "departure_time": dep_time,
                    "alpha": alpha,
                    "status": "integration_failed",
                    "dv_departure": dv_dep,
                    "dv_insertion": None,
                    "departure_orbit_name": dep_name,
                    "arrival_orbit_name": arr_name,
                    "departure_time_index": dep_idx,
                }
            )
            if progress_queue is not None:
                progress_queue.put(1)
            continue

        # GEO 到达判断
        collision, body, col_idx = _check_collision(
            states, mu, collision_earth_radius, collision_moon_radius
        )
        crossed, cross_idx = _detect_sphere_crossing(states, r_geo, earth_center)
        min_dist, closest_idx = _closest_approach_to_sphere(states, r_geo, earth_center)
        dv_ins = float(
            np.linalg.norm(
                states[closest_idx, 3:] - _geo_circular_velocity(states[closest_idx, :3], mu)
            )
        )

        if crossed:
            dv_ins = float(
                np.linalg.norm(
                    states[cross_idx, 3:] - _geo_circular_velocity(states[cross_idx, :3], mu)
                )
            )
            transfer_time = float(times[cross_idx])
            status = "success"
        elif min_dist < geo_threshold:
            transfer_time = float(times[closest_idx])
            status = "success"
        else:
            transfer_time = float(times[-1])
            status = "no_crossing"
        if collision:
            status = "collision"

        results.append(
            {
                "success": True,
                "departure_state": dep_state,
                "departure_time": dep_time,
                "alpha": alpha,
                "transfer_time": transfer_time,
                "geo_crossing_found": crossed,
                "geo_crossing_idx": int(cross_idx) if crossed else None,
                "min_distance_to_geo": min_dist,
                "closest_geo_idx": int(closest_idx),
                "dv_departure": dv_dep,
                "dv_insertion": dv_ins,
                "collision_found": collision,
                "collision_body": body,
                "collision_idx": col_idx,
                "status": status,
                "departure_orbit_name": dep_name,
                "arrival_orbit_name": arr_name,
                "departure_time_index": dep_idx,
            }
        )
        if progress_queue is not None:
            progress_queue.put(1)

    return results


def _check_collision(states, mu, earth_r, moon_r):
    pos = states[:, :3]
    d_earth = np.linalg.norm(pos - np.array([-mu, 0.0, 0.0]), axis=1)
    d_moon = np.linalg.norm(pos - np.array([1.0 - mu, 0.0, 0.0]), axis=1)
    e_col = np.where(d_earth < earth_r)[0]
    m_col = np.where(d_moon < moon_r)[0]
    if len(e_col) > 0:
        return True, "earth", int(e_col[0])
    if len(m_col) > 0:
        return True, "moon", int(m_col[0])
    return False, None, -1


# ── GeoTransferSearch 类 ──


class GeoTransferSearch(TransferSearch):
    """DRO → GEO 球面转移搜索。

    继承 TransferSearch，复用采样、积分、碰撞检测、并行调度框架。
    仅重写到达判断逻辑：GEO 球面穿越代替 RO 轨道相交。
    """

    def __init__(self, dynamics, geo_threshold: float = 100.0 / DU):
        super().__init__(dynamics, name="GeoTransferSearch")
        self.geo_threshold = geo_threshold

    def search(
        self,
        *,
        departure_orbit,
        alpha_min: float = 0.5,
        alpha_max: float = 2.5,
        n_alpha: int = 100,
        n_departure: int = 200,
        max_transfer_time: float = 100.0 / 4.34811305,
        intersection_threshold: float = 0.001,
        min_distance_threshold: float = 100.0 / DU,
        collision_earth_radius: float = 200.0 / DU,
        collision_moon_radius: float = 100.0 / DU,
        integration_dt: float = 1.0 / (24.0 * 4.34811305),
        geo_threshold: Optional[float] = None,
        n_workers: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """执行 GEO 球面搜索。"""
        if geo_threshold is not None:
            self.geo_threshold = geo_threshold

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.n_alpha = n_alpha
        self.n_departure = n_departure
        self.max_transfer_time = max_transfer_time
        self.intersection_threshold = intersection_threshold
        self.min_distance_threshold = min_distance_threshold
        self.collision_earth_radius = collision_earth_radius
        self.collision_moon_radius = collision_moon_radius
        self.integration_dt = integration_dt

        self._departure_orbit = departure_orbit

        if n_workers is None:
            n_workers = multiprocessing.cpu_count()

        dep_states, dep_times = self._sample_departure_points(departure_orbit)
        dep_name = getattr(departure_orbit, "name", "DRO")

        if verbose:
            print(f"  GEO 半径: {R_GEO:.6f} DU ({R_GEO * DU:.0f} km)")
            print(f"  α: [{alpha_min}, {alpha_max}], n={n_alpha}")
            print(f"  出发点: {n_departure}")
            print(f"  进程数: {n_workers}")

        if n_workers <= 1:
            results = self._grid_search_sequential(
                dep_states,
                dep_times,
                dep_name,
                verbose,
            )
        else:
            results = self._grid_search_parallel_geo(
                dep_states,
                dep_times,
                dep_name,
                verbose,
                n_workers,
            )

        self._search_results = results
        return results

    def _grid_search_sequential(
        self,
        dep_states,
        dep_times,
        dep_name,
        verbose,
    ) -> List[Dict[str, Any]]:
        """串行搜索。复用父类 _search_single_departure 的调用模式。"""
        n_alpha = int(self.n_alpha or 100)
        total = len(dep_states) * n_alpha
        pbar = (
            self._open_search_progress_bar(total, "GEO 网格搜索") if verbose and total > 0 else None
        )

        all_results = []
        try:
            for i, (dep_state, dep_time) in enumerate(zip(dep_states, dep_times)):
                results = self._search_single_departure(
                    dep_state,
                    dep_time,
                    None,
                    verbose=(verbose and pbar is None),
                    pbar=pbar,
                    departure_index=i,
                )
                for r in results:
                    r["departure_orbit_name"] = dep_name
                    r["arrival_orbit_name"] = "GEO"
                    r["departure_time_index"] = i
                all_results.extend(results)
        finally:
            if pbar is not None:
                pbar.close()
        return all_results

    def _grid_search_parallel_geo(
        self,
        dep_states,
        dep_times,
        dep_name,
        verbose,
        n_workers,
    ) -> List[Dict[str, Any]]:
        """并行搜索，使用 GEO 专用 worker。与 TransferSearch._grid_search_parallel_processes 结构一致。"""
        n_alpha = int(self.n_alpha or 100)
        total_steps = len(dep_states) * n_alpha

        dyn = self.dynamics
        pack_base = (
            self.mu,
            self.geo_threshold,
            float(self.alpha_min or 0.5),
            float(self.alpha_max or 2.5),
            n_alpha,
            self.n_departure,
            float(self.max_transfer_time or 0),
            float(self.intersection_threshold or 0.001),
            float(self.min_distance_threshold or 0),
            float(self.collision_earth_radius or 0),
            float(self.collision_moon_radius or 0),
            float(self.integration_dt or 0),
            str(dyn.integrator),
            float(dyn.rtol),
            float(dyn.atol),
            float(dyn.max_step),
            dep_name,
            "GEO",
        )

        pbar = tqdm(total=total_steps, desc="GEO 并行搜索", dynamic_ncols=True) if verbose else None
        progress_manager = multiprocessing.Manager()
        progress_queue = progress_manager.Queue()
        poll_stop = threading.Event()

        if pbar is not None:

            def _poll():
                while not poll_stop.is_set():
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

        all_results = []
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {}
                for i, (dep_state, dep_time) in enumerate(zip(dep_states, dep_times)):
                    packed = (
                        (i, np.asarray(dep_state, dtype=float), float(dep_time))
                        + pack_base
                        + (progress_queue,)
                    )
                    fut = executor.submit(_geo_search_worker, packed)
                    futures[fut] = i
                for fut in as_completed(futures):
                    try:
                        all_results.extend(fut.result())
                    except Exception as e:
                        if verbose:
                            tqdm.write(f"    出发点 {futures[fut]} 处理失败: {e}")
        finally:
            poll_stop.set()
            poll_thread.join(timeout=5)
            if pbar is not None:
                pbar.close()
            progress_manager.shutdown()

        return all_results

    def _search_single_departure(
        self,
        departure_state,
        departure_time,
        arrival_orbit=None,
        verbose=False,
        pbar=None,
        departure_index=None,
        progress_queue=None,
    ) -> List[Dict[str, Any]]:
        """GEO 到达版搜索。"""
        results = []
        a0, a1, na = self.alpha_min, self.alpha_max, self.n_alpha
        mtt, idt = self.max_transfer_time, self.integration_dt
        if any(v is None for v in [a0, a1, na, mtt, idt]):
            raise ValueError("搜索参数未设置")
        alpha_grid = np.linspace(a0, a1, int(na))
        mu = self.mu
        earth_center = np.array([-mu, 0.0, 0.0])
        r_geo = R_GEO

        for i_alpha, alpha in enumerate(alpha_grid, start=1):
            try:
                new_vel = self._compute_departure_velocity(departure_state, alpha)
                dv_dep = float(np.linalg.norm(new_vel - departure_state[3:6]))
                x0 = np.concatenate([departure_state[:3], new_vel])
                try:
                    states, times = self._forward_integrate(x0, mtt, idt)
                except Exception:
                    results.append(
                        {
                            "success": False,
                            "departure_state": departure_state,
                            "departure_time": departure_time,
                            "alpha": alpha,
                            "status": "integration_failed",
                            "dv_departure": dv_dep,
                            "dv_insertion": None,
                        }
                    )
                    continue

                collision, body, col_idx = self._check_collision(states)
                crossed, cross_idx = _detect_sphere_crossing(states, r_geo, earth_center)
                min_dist, closest_idx = _closest_approach_to_sphere(states, r_geo, earth_center)
                dv_ins = float(
                    np.linalg.norm(
                        states[closest_idx, 3:]
                        - _geo_circular_velocity(states[closest_idx, :3], mu)
                    )
                )
                if crossed:
                    dv_ins = float(
                        np.linalg.norm(
                            states[cross_idx, 3:]
                            - _geo_circular_velocity(states[cross_idx, :3], mu)
                        )
                    )
                    transfer_time = float(times[cross_idx])
                    status = "success"
                elif min_dist < self.geo_threshold:
                    transfer_time = float(times[closest_idx])
                    status = "success"
                else:
                    transfer_time = float(times[-1])
                    status = "no_crossing"
                if collision:
                    status = "collision"

                results.append(
                    {
                        "success": True,
                        "departure_state": departure_state,
                        "departure_time": departure_time,
                        "alpha": alpha,
                        "transfer_time": transfer_time,
                        "geo_crossing_found": crossed,
                        "geo_crossing_idx": int(cross_idx) if crossed else None,
                        "min_distance_to_geo": min_dist,
                        "closest_geo_idx": int(closest_idx),
                        "dv_departure": dv_dep,
                        "dv_insertion": dv_ins,
                        "collision_found": collision,
                        "collision_body": body,
                        "collision_idx": col_idx,
                        "status": status,
                    }
                )
            finally:
                if pbar is not None:
                    pbar.update(1)
                    if departure_index is not None:
                        pbar.set_postfix_str(f"dep={departure_index} α={alpha:.4f}")
                if progress_queue is not None:
                    progress_queue.put(1)

        return results

    def _is_feasible(self, result: Dict[str, Any]) -> bool:
        if result.get("collision_found", False):
            return False
        return result.get("status") == "success"

    def get_feasible_results(self) -> List[Dict[str, Any]]:
        if self._search_results is None:
            return []
        return [r for r in self._search_results if self._is_feasible(r)]
