"""
平面DRO到RO转移轨道搜索模块

实现论文Cui et al. (2025)中的"搜索-优化"两步法的搜索阶段。
专门用于平面转移轨道设计，搜索变量: 出发点位置、α(切向速度比)

使用方式:
    # 方式1: 使用新架构 (推荐)
    transfer = DROTransferSearch(system, dynamics)
    transfer.set_departure_orbit(dro_orbit).set_arrival_orbit(ro_orbit)
    transfer.configure_search(alpha_range=(0.5, 2.5), n_departure=200)
    results = transfer.search()
    optimized = transfer.optimize(results[0])

    # 方式2: 使用传统方式 (保持向后兼容)
    config = TransferSearchConfig(alpha_min=0.5, alpha_max=2.5, n_alpha=101)
    searcher = DROROTransferSearch(system, dynamics, config)
    results = searcher.grid_search(departure_orbit, arrival_orbit)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import warnings

from ..core.orbit import Orbit
from ..core.dynamics import CR3BP_Dynamics
from ..core.system import CR3BP_System

from .transfer_base import (
    BaseTransfer,
    SearchConfig,
    SearchResult,
    OptimizationConfig,
    OptimizationResult,
    TransferType,
)


class DROTransferSearch(BaseTransfer):
    """DRO到RO平面转移轨道搜索算法

    实现论文Section III.A的搜索阶段算法:
    1. 从出发点轨道等时间间隔采样
    2. 对每个出发点，网格化搜索α
    3. 前向积分获取转移轨迹
    4. 筛选与目标轨道相交或距离局部最小的候选解

    使用方式:
        transfer = DROTransferSearch(system, dynamics)
        transfer.set_departure_orbit(dro_orbit)
        transfer.set_arrival_orbit(ro_orbit)
        transfer.configure_search(alpha_min=0.5, alpha_max=2.5, n_alpha=101)
        results = transfer.search()
    """

    def __init__(
        self,
        system: CR3BP_System,
        dynamics: CR3BP_Dynamics,
        name: str = "DROTransferSearch",
    ):
        super().__init__(system, dynamics, name)
        self._verbose = True
        self._n_workers = None

        self.alpha_min = 0.5
        self.alpha_max = 2.5
        self.n_alpha = 101
        self.n_departure = 200
        self.max_transfer_time = 15.0
        self.intersection_threshold = 0.001
        self.min_distance_threshold = 0.05
        self.collision_earth_radius = 1.0 - 0.999
        self.collision_moon_radius = 0.999
        self.integration_dt = 0.01

    def set_verbose(self, verbose: bool) -> "DROTransferSearch":
        """设置是否输出详细信息"""
        self._verbose = verbose
        return self

    def set_n_workers(self, n_workers: int) -> "DROTransferSearch":
        """设置并行worker数量"""
        self._n_workers = n_workers
        return self

    def configure_search(self, **kwargs) -> "DROTransferSearch":
        """配置搜索参数（向后兼容，推荐直接设置实例属性）"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def search(self, **kwargs) -> List[SearchResult]:
        """执行网格搜索

        参数:
            **kwargs: 搜索参数，可选:
                - verbose: 是否输出详细信息
                - n_workers: 并行worker数量

        返回:
            搜索结果列表
        """
        if self._departure_orbit is None or self._arrival_orbit is None:
            raise ValueError("必须先设置departure_orbit和arrival_orbit")

        verbose = kwargs.get("verbose", self._verbose)
        n_workers = kwargs.get("n_workers", self._n_workers)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"DRO-RO转移轨道网格搜索")
            print(f"{'=' * 60}")
            print(f"出发点: {self._departure_orbit}")
            print(f"目标: {self._arrival_orbit}")
            print(f"α范围: [{self.alpha_min}, {self.alpha_max}], n={self.n_alpha}")
            print(f"出发点数量: {self.n_departure}")
            print(f"{'=' * 60}\n")

        results = self._grid_search(
            self._departure_orbit,
            self._arrival_orbit,
            verbose,
            n_workers,
        )

        self._search_results = results

        if verbose:
            feasible = [r for r in results if r.is_feasible]
            print(f"\n{'=' * 60}")
            print(f"搜索完成")
            print(f"  总候选解: {len(results)}")
            print(f"  可行解: {len(feasible)}")
            print(f"{'=' * 60}")

        return results

    def optimize(self, initial_guess: Optional[SearchResult] = None) -> OptimizationResult:
        """执行优化

        参数:
            initial_guess: 优化初始猜测（通常来自搜索结果）

        返回:
            优化结果
        """
        from .transfer_optimization import DROTRONLPOptimizer, NLPOptimizationVariables

        if self._departure_orbit is None or self._arrival_orbit is None:
            raise ValueError("必须先设置departure_orbit和arrival_orbit")

        if initial_guess is None:
            if self._search_results:
                feasible = self.get_feasible_results()
                if feasible:
                    initial_guess = feasible[0]
                else:
                    initial_guess = self._search_results[0]
            else:
                raise ValueError("必须提供initial_guess或先运行search()")

        config = self._optimization_config or OptimizationConfig()

        optimizer = DROTRONLPOptimizer(
            system=self.system,
            dynamics=self.dynamics,
            departure_orbit=self._departure_orbit,
            arrival_orbit=self._arrival_orbit,
            departure_state=initial_guess.departure_state,
        )

        nlp_vars = NLPOptimizationVariables(
            alpha=initial_guess.alpha,
            transfer_time=initial_guess.transfer_time or config.transfer_time_range[1] / 2,
            t_ins=0.0,
        )

        result = optimizer.optimize(
            initial_guess=nlp_vars,
            alpha_range=config.alpha_range,
            transfer_time_range=config.transfer_time_range,
            t_ins_range=config.t_ins_range,
            verbose=self._verbose,
        )

        self._optimized_result = result
        return result

    def _grid_search(
        self,
        departure_orbit: Orbit,
        arrival_orbit: Orbit,
        verbose: bool,
        n_workers: Optional[int],
    ) -> List[SearchResult]:
        """执行网格搜索的内部方法"""
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor, as_completed

        dep_name = getattr(departure_orbit, "name", "unknown")
        arr_name = getattr(arrival_orbit, "name", "unknown")

        departure_states, departure_times = self._sample_departure_points(departure_orbit)
        total_departures = len(departure_states)

        if n_workers is None:
            n_workers = multiprocessing.cpu_count()

        if n_workers == 1:
            return self._grid_search_sequential(
                departure_states,
                departure_times,
                arrival_orbit,
                dep_name,
                arr_name,
                verbose,
            )
        else:
            return self._grid_search_parallel(
                departure_states,
                departure_times,
                arrival_orbit,
                dep_name,
                arr_name,
                verbose,
                n_workers,
            )

    def _sample_departure_points(
        self, departure_orbit: Orbit
    ) -> Tuple[np.ndarray, np.ndarray]:
        """从轨道等时间间隔采样出发点"""
        n = self.n_departure
        times = np.linspace(0, departure_orbit.period, n, endpoint=False)
        states = np.array([departure_orbit.interpolate_at_time(t) for t in times])
        return states, times

    def _grid_search_sequential(
        self,
        departure_states: np.ndarray,
        departure_times: np.ndarray,
        arrival_orbit: Orbit,
        dep_name: str,
        arr_name: str,
        verbose: bool,
    ) -> List[SearchResult]:
        """串行网格搜索"""
        all_results = []
        total_departures = len(departure_states)

        for i, (dep_state, dep_time) in enumerate(zip(departure_states, departure_times)):
            if verbose:
                pct = (i + 1) / total_departures * 100
                print(f"  进度: {i + 1}/{total_departures} ({pct:.1f}%)")

            results = self._search_single_departure(dep_state, dep_time, arrival_orbit)
            for r in results:
                r.departure_orbit_name = dep_name
                r.arrival_orbit_name = arr_name
                r.departure_time_index = i
            all_results.extend(results)

        return all_results

    def _grid_search_parallel(
        self,
        departure_states: np.ndarray,
        departure_times: np.ndarray,
        arrival_orbit: Orbit,
        dep_name: str,
        arr_name: str,
        verbose: bool,
        n_workers: int,
    ) -> List[SearchResult]:
        """并行网格搜索"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total_departures = len(departure_states)
        all_results = []
        completed = 0

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    self._search_single_departure,
                    dep_state,
                    dep_time,
                    arrival_orbit,
                ): i
                for i, (dep_state, dep_time) in enumerate(zip(departure_states, departure_times))
            }

            for future in as_completed(futures):
                completed += 1
                if verbose:
                    pct = completed / total_departures * 100
                    print(f"  进度: {completed}/{total_departures} ({pct:.1f}%)")

                try:
                    results = future.result()
                    for r in results:
                        r.departure_orbit_name = dep_name
                        r.arrival_orbit_name = arr_name
                        r.departure_time_index = futures[future]
                    all_results.extend(results)
                except Exception as e:
                    if verbose:
                        print(f"    出发点 {futures[future]} 处理失败: {e}")

        return all_results

    def _search_single_departure(
        self,
        departure_state: np.ndarray,
        departure_time: float,
        arrival_orbit: Orbit,
    ) -> List[SearchResult]:
        """对单个出发点搜索α网格"""
        results = []

        alpha_grid = np.linspace(self.alpha_min, self.alpha_max, self.n_alpha)

        for alpha in alpha_grid:
            new_vel = self._compute_departure_velocity(departure_state, alpha)

            initial_state = np.concatenate(
                [
                    departure_state[:3],
                    new_vel,
                ]
            )

            try:
                traj_states, traj_times = self._forward_integrate(
                    initial_state, self.max_transfer_time, self.integration_dt
                )
            except Exception:
                result = SearchResult(
                    success=False,
                    departure_state=departure_state,
                    departure_time=departure_time,
                    alpha=alpha,
                    status="integration_failed",
                )
                results.append(result)
                continue

            collision, body, col_idx = self._check_collision(traj_states)
            min_dist, min_idx = self._compute_min_distance(traj_states, arrival_orbit)
            intersection, int_point, int_idx = self._detect_intersection(
                traj_states, arrival_orbit, self.intersection_threshold
            )
            local_min, local_min_dist, local_min_idx = self._detect_local_minimum(
                traj_states, arrival_orbit
            )

            result = SearchResult(
                success=True,
                departure_state=departure_state,
                departure_time=departure_time,
                alpha=alpha,
                transfer_trajectory=traj_states,
                transfer_times=traj_times,
                transfer_time=traj_times[-1],
                min_distance=min_dist,
                min_distance_idx=min_idx,
                intersection_found=intersection,
                intersection_point=int_point,
                intersection_idx=int_idx,
                local_minimum_found=local_min,
                local_minimum_distance=local_min_dist,
                local_minimum_idx=local_min_idx,
                collision_found=collision,
                collision_body=body,
                collision_idx=col_idx,
            )

            if collision:
                result.status = "collision"
            elif intersection:
                result.status = "success"
            elif min_dist < self.min_distance_threshold:
                result.status = "success"
            else:
                result.status = "no_intersection"

            results.append(result)

        return results

    def _compute_departure_velocity(self, orbit_state: np.ndarray, alpha: float) -> np.ndarray:
        """计算出发点速度扰动 (平面)"""
        pos = orbit_state[:3]
        vel = orbit_state[3:]

        r_xy = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
        if r_xy < 1e-10:
            warnings.warn("位置靠近原点，使用原始速度")
            return vel.copy()

        tangential = np.array([-pos[1], pos[0], 0.0]) / r_xy
        radial = pos / np.linalg.norm(pos)

        v_radial_comp = np.dot(vel, radial)
        v_tangential_comp = np.dot(vel, tangential)

        new_vel = v_radial_comp * radial + alpha * v_tangential_comp * tangential
        return new_vel

    def _forward_integrate(
        self,
        initial_state: np.ndarray,
        transfer_time: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """前向积分转移轨迹"""
        n_steps = max(int(transfer_time / dt) + 1, 100)
        t_eval = np.linspace(0, transfer_time, n_steps)

        result = self.dynamics.propagate(
            initial_state=initial_state,
            t_span=[0, transfer_time],
            t_eval=t_eval,
            with_stm=False,
        )

        return result["states"], result["time"]

    def _compute_min_distance(
        self, trajectory_states: np.ndarray, arrival_orbit: Orbit
    ) -> Tuple[float, int]:
        """计算轨迹到目标轨道的最小距离"""
        traj_positions = trajectory_states[:, :3]
        orbit_positions = arrival_orbit.states[:, :3]

        diff = traj_positions[:, np.newaxis, :] - orbit_positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))

        flat_distances = distances.flatten()
        min_flat_idx = np.argmin(flat_distances)
        min_distance = flat_distances[min_flat_idx]

        n_orbit = len(orbit_positions)
        step_idx = min_flat_idx // n_orbit

        return min_distance, step_idx

    def _detect_intersection(
        self,
        trajectory_states: np.ndarray,
        arrival_orbit: Orbit,
        threshold: float,
    ) -> Tuple[bool, Optional[np.ndarray], int]:
        """检测轨迹是否与目标轨道相交"""
        min_dist, step_idx = self._compute_min_distance(trajectory_states, arrival_orbit)

        if min_dist < threshold:
            return True, trajectory_states[step_idx], step_idx

        return False, None, -1

    def _detect_local_minimum(
        self,
        trajectory_states: np.ndarray,
        arrival_orbit: Orbit,
    ) -> Tuple[bool, float, int]:
        """检测轨迹到目标轨道的距离是否出现局部最小"""
        traj_positions = trajectory_states[:, :3]
        orbit_positions = arrival_orbit.states[:, :3]

        diff = traj_positions[:, np.newaxis, :] - orbit_positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        min_distances = np.min(distances, axis=1)

        local_mins = []
        for i in range(1, len(min_distances) - 1):
            if min_distances[i + 1] > min_distances[i] and min_distances[i - 1] > min_distances[i]:
                local_mins.append((i, min_distances[i]))

        if local_mins:
            best = min(local_mins, key=lambda x: x[1])
            return True, best[1], best[0]

        return False, np.inf, -1

    def _check_collision(
        self,
        trajectory_states: np.ndarray,
    ) -> Tuple[bool, Optional[str], int]:
        """检测轨迹是否与地球或月球碰撞"""
        positions = trajectory_states[:, :3]

        earth_center = np.array([-self.mu, 0.0, 0.0])
        moon_center = np.array([1.0 - self.mu, 0.0, 0.0])

        dist_earth = np.linalg.norm(positions - earth_center, axis=1)
        dist_moon = np.linalg.norm(positions - moon_center, axis=1)

        earth_collision_idx = np.where(dist_earth < self.collision_earth_radius)[0]
        moon_collision_idx = np.where(dist_moon < self.collision_moon_radius)[0]

        if len(earth_collision_idx) > 0:
            return True, "earth", int(earth_collision_idx[0])
        if len(moon_collision_idx) > 0:
            return True, "moon", int(moon_collision_idx[0])

        return False, None, -1


# 向后兼容别名
DROROTransferSearch = DROTransferSearch


def _process_departure_worker(
    idx: int,
    dep_state: np.ndarray,
    dep_time: float,
    arrival_states: np.ndarray,
    arrival_times: np.ndarray,
    arrival_period: float,
    mu: float,
    config: SearchConfig,
    dep_name: str,
    arr_name: str,
) -> List[SearchResult]:
    """Worker function for parallel departure point processing.

    This is a module-level function to ensure pickling works on Windows (spawn mode).
    """
    arrival_orbit = Orbit(states=arrival_states, times=arrival_times)
    arrival_orbit.period = arrival_period

    searcher = DROROTransferSearch(
        system=CR3BP_System(mu=mu, primary="earth", secondary="moon"),
        dynamics=CR3BP_Dynamics(system=CR3BP_System(mu=mu, primary="earth", secondary="moon")),
    )

    searcher.configure_search(
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
        n_alpha=config.n_alpha,
        n_departure=config.n_departure,
        max_transfer_time=config.max_transfer_time,
        intersection_threshold=config.intersection_threshold,
        min_distance_threshold=config.min_distance_threshold,
        collision_earth_radius=config.collision_earth_radius,
        collision_moon_radius=config.collision_moon_radius,
        integration_dt=config.integration_dt,
    )

    results = searcher._search_single_departure(dep_state, dep_time, arrival_orbit)
    for r in results:
        r.departure_orbit_name = dep_name
        r.arrival_orbit_name = arr_name
        r.departure_time_index = idx
    return results


def load_orbit_from_json(filepath: str) -> Orbit:
    """从JSON文件加载轨道数据

    参数:
        filepath: JSON文件路径

    返回:
        Orbit对象
    """
    import json

    with open(filepath, "r") as f:
        data = json.load(f)

    states = np.array(data["states"])
    times = np.array(data["times"])

    orbit = Orbit(states=states, times=times)

    if "orbit_type" in data:
        orbit.metadata["orbit_type"] = data["orbit_type"]
    if "period_ratio" in data:
        orbit.metadata["period_ratio"] = data["period_ratio"]

    return orbit


def save_search_results(
    results: List[SearchResult],
    filepath: str,
) -> None:
    """保存搜索结果到JSON文件

    参数:
        results: 搜索结果列表
        filepath: 输出文件路径
    """
    import json

    output = []
    for r in results:
        result_dict = {
            "departure_orbit_name": r.departure_orbit_name,
            "arrival_orbit_name": r.arrival_orbit_name,
            "departure_time_index": r.departure_time_index,
            "departure_time": float(r.departure_time),
            "alpha": float(r.alpha),
            "transfer_time": float(r.transfer_time),
            "intersection_found": r.intersection_found,
            "min_distance": float(r.min_distance),
            "local_minimum_found": r.local_minimum_found,
            "collision_found": r.collision_found,
            "status": r.status,
            "is_feasible": r.is_feasible,
        }
        output.append(result_dict)

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
