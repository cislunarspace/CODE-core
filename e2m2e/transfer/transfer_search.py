"""
平面DRO到RO转移轨道搜索模块

实现论文Cui et al. (2025)中的"搜索-优化"两步法的搜索阶段。
专门用于平面转移轨道设计，搜索变量: 出发点位置、α(切向速度比)

使用方式:
    transfer = DROTransferSearch(system, dynamics)
    transfer.set_departure_orbit(dro_orbit).set_arrival_orbit(ro_orbit)
    transfer.alpha_min = 0.5
    transfer.alpha_max = 2.5
    transfer.n_alpha = 101
    # ... 设置其他参数
    results = transfer.search()
"""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Optional, Dict, Any
import warnings

import numpy as np
from tqdm.auto import tqdm


class _AggregatePbarWithSlot:
    """共享总进度：转发 update/postfix，加锁并加槽位前缀。"""

    __slots__ = ("_inner", "_lock", "_slot")

    def __init__(self, inner: Any, lock: Optional[threading.Lock], slot: int) -> None:
        self._inner = inner
        self._lock = lock
        self._slot = slot

    def update(self, n: int = 1) -> None:
        if self._lock is not None:
            with self._lock:
                self._inner.update(n)
        else:
            self._inner.update(n)

    def set_postfix_str(self, s: str, refresh: bool = True) -> None:
        merged = f"W{self._slot} {s}"
        if self._lock is not None:
            with self._lock:
                self._inner.set_postfix_str(merged, refresh=refresh)
        else:
            self._inner.set_postfix_str(merged, refresh=refresh)

from ..core.orbit import Orbit
from ..core.dynamics import CR3BP_Dynamics
from ..core.system import CR3BP_System

from .transfer_base import (
    BaseTransfer,
    TransferType,
)
from .transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
    NLPOptimizationResult,
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
        super().__init__(system, dynamics)
        self.name = name
        self._verbose = True
        self._n_workers = None

        # α (切向速度比) 搜索范围
        # 推荐值: alpha_min ∈ (0, 1.0], alpha_max ∈ [1.0, 3.0]
        # 约束: 0 < alpha_min < alpha_max
        self.alpha_min = None
        self.alpha_max = None

        # α 方向网格点数
        # 推荐值: n_alpha ∈ [51, 2001], 典型值 101 或 201
        # 约束: n_alpha >= 2
        self.n_alpha = None

        # 出发点采样数量
        # 推荐值: n_departure ∈ [50, 500], 典型值 200
        # 约束: n_departure >= 2
        self.n_departure = None

        # 最大转移时间 (CR3BP 无量纲时间单位)
        # 推荐值: max_transfer_time ∈ [5.0, 30.0], 典型值 15.0
        # 约束: max_transfer_time > 0
        self.max_transfer_time = None

        # 相交判定阈值 (无量纲距离)
        # 推荐值: intersection_threshold ∈ [1e-4, 1e-2], 典型值 0.001
        # 约束: intersection_threshold > 0
        self.intersection_threshold = None

        # 候选解最小距离阈值 (无量纲距离)
        # 推荐值: min_distance_threshold ∈ [0.01, 0.1], 典型值 0.05
        # 约束: min_distance_threshold > 0
        self.min_distance_threshold = None

        # 地球碰撞检测半径 (无量纲距离)
        # 推荐值: 200 km ≈ 0.0005 (相对于地月距离 384405 km)
        # 约束: collision_earth_radius > 0
        self.collision_earth_radius = None

        # 月球碰撞检测半径 (无量纲距离)
        # 推荐值: 100 km ≈ 0.00026
        # 约束: collision_moon_radius > 0
        self.collision_moon_radius = None

        # 积分时间步长 (CR3BP 无量纲时间)
        # 推荐值: integration_dt ∈ [1e-4, 0.1], 典型值 0.01 //TODO 这里的典型值需要修改，这里的单位是无量纲量，所以不能以0.01为单位，而应该以转换之后的10s或者60s作为典型值
        # 约束: integration_dt > 0
        self.integration_dt = None

        # 优化参数
        # alpha 搜索范围
        # 推荐值: (0.5, 2.5)
        self.alpha_range = None

        # 转移时间范围
        # 推荐值: (1.0, 30.0)
        self.transfer_time_range = None

        # 插入时间范围
        # 推荐值: (0.0, 10.0)
        self.t_ins_range = None

        # 速度平行性容差
        # 推荐值: 1e-6
        self.velocity_angle_tolerance = None

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

    def search(self, **kwargs) -> List[Dict[str, Any]]:
        """执行网格搜索

        参数:
            **kwargs: 搜索参数，可选:
                - verbose: 是否输出详细信息（含进度）
                - n_workers: 并行 worker 数量

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
            feasible = [r for r in results if self._is_feasible(r)]
            print(f"\n{'=' * 60}")
            print(f"搜索完成")
            print(f"  总候选解: {len(results)}")
            print(f"  可行解: {len(feasible)}")
            print(f"{'=' * 60}")

        return results

    def optimize(self, initial_guess: Optional[Dict[str, Any]] = None) -> "NLPOptimizationResult":
        """执行优化

        参数:
            initial_guess: 优化初始猜测（通常来自搜索结果）

        返回:
            优化结果
        """
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

        optimizer = DROTRONLPOptimizer(
            system=self.system,
            dynamics=self.dynamics,
            departure_orbit=self._departure_orbit,
            arrival_orbit=self._arrival_orbit,
            departure_state=initial_guess["departure_state"],
        )

        transfer_time = self.transfer_time_range[1] / 2 if self.transfer_time_range else 15.0

        nlp_vars = NLPOptimizationVariables(
            alpha=initial_guess.get("alpha", 1.0),
            transfer_time=initial_guess.get("transfer_time") or transfer_time,
            t_ins=0.0,
        )

        result = optimizer.optimize(
            initial_guess=nlp_vars,
            alpha_range=self.alpha_range or (0.5, 2.5),
            transfer_time_range=self.transfer_time_range or (1.0, 30.0),
            t_ins_range=self.t_ins_range or (0.0, 10.0),
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
    ) -> List[Dict[str, Any]]:
        """执行网格搜索的内部方法"""
        # 获取出发轨道和目标轨道的名称
        dep_name = getattr(departure_orbit, "name", "unknown")
        arr_name = getattr(arrival_orbit, "name", "unknown")

        # //TODO 这是什么？
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

    def _sample_departure_points(self, departure_orbit: Orbit) -> Tuple[np.ndarray, np.ndarray]:
        """从星历中均匀下采样出发点：直接使用 ``Orbit.states`` / ``Orbit.times`` 已有行，不积分。

        要求 ``n_departure <= len(times)``，否则报错。
        """
        times = departure_orbit.times
        states = departure_orbit.states
        n_pts = len(times)
        if n_pts == 0:
            raise ValueError("出发轨道无数据点")
        n = int(self.n_departure)
        if n <= 0:
            raise ValueError("n_departure 须为正整数")
        if n > n_pts:
            raise ValueError(
                f"n_departure（{n}）不能大于出发轨道星历点数（{n_pts}），请减小 n_departure 或增加星历密度"
            )

        n_sample = n
        if n_sample == 1:
            idx = np.array([0], dtype=int)
        else:
            idx = (np.arange(n_sample, dtype=float) * (n_pts - 1) / (n_sample - 1)).round().astype(
                int
            )

        return states[idx].copy(), times[idx].copy()

    def _open_search_progress_bar(self, total: int, desc: str) -> Optional[Any]:
        """``total <= 0`` 时返回 None。"""
        if total <= 0:
            return None
        return tqdm(
            total=total,
            desc=desc,
            unit="it",
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.2,
        )

    @staticmethod
    def _use_multiline_worker_tqdm(n_workers: int) -> bool:
        """非 TTY 用单行总进度；否则可分槽多行。环境变量 ``E2M2E_TQDM_MULTILINE`` 可强制开/关；槽数>32 只用单行。"""
        env = os.environ.get("E2M2E_TQDM_MULTILINE", "").strip().lower()
        if env in ("0", "false", "no", "off"):
            return False
        if env in ("1", "true", "yes", "on"):
            return True
        if n_workers > 32:
            return False
        return sys.stderr.isatty()

    @staticmethod
    def _reset_tqdm_bar(bar: Any, total: int) -> None:
        """复用进度条时重置 total。"""
        if hasattr(bar, "reset"):
            bar.reset(total=total)
        else:
            bar.n = 0
            bar.total = total

    def _open_parallel_worker_progress_bars(self, n_workers: int, n_alpha: int) -> List[Any]:
        """每槽一行，每行 ``n_alpha`` 步。"""
        return [
            tqdm(
                total=n_alpha,
                position=i,
                desc=f"W{i}",
                leave=True,
                unit="α",
                file=sys.stderr,
                dynamic_ncols=True,
                mininterval=0.1,
            )
            for i in range(n_workers)
        ]

    def _run_departure_with_worker_slot(
        self,
        slot_queue: "queue.Queue[int]",
        n_alpha: int,
        departure_index: int,
        departure_state: np.ndarray,
        departure_time: float,
        arrival_orbit: Orbit,
        worker_bars: Optional[List[Any]],
        aggregate_pbar: Optional[Any],
        aggregate_lock: Optional[threading.Lock],
    ) -> List[Dict[str, Any]]:
        """取槽 → 跑单出发点 α 网格 → 还槽。``worker_bars`` 与 ``aggregate_pbar`` 二选一。"""
        slot = slot_queue.get()
        try:
            if worker_bars is not None:
                bar = worker_bars[slot]
                self._reset_tqdm_bar(bar, n_alpha)
                bar.set_description_str(f"W{slot} dep={departure_index}", refresh=False)
                pbar: Any = bar
            elif aggregate_pbar is not None:
                pbar = _AggregatePbarWithSlot(aggregate_pbar, aggregate_lock, slot)
            else:
                raise RuntimeError("worker_bars 与 aggregate_pbar 至少传入其一")
            return self._search_single_departure(
                departure_state,
                departure_time,
                arrival_orbit,
                verbose=False,
                pbar=pbar,
                departure_index=departure_index,
            )
        finally:
            slot_queue.put(slot)

    def _grid_search_sequential(
        self,
        departure_states: np.ndarray,
        departure_times: np.ndarray,
        arrival_orbit: Orbit,
        dep_name: str,
        arr_name: str,
        verbose: bool,
    ) -> List[Dict[str, Any]]:
        """串行网格搜索"""
        all_results = []
        total_departures = len(departure_states)
        n_alpha = int(self.n_alpha)
        total_steps = total_departures * n_alpha
        pbar = None
        if verbose and total_steps > 0:
            pbar = self._open_search_progress_bar(total_steps, "网格搜索")

        if verbose and total_steps <= 0:
            print(f"  总迭代步数: {total_steps}（出发点 × α），无进度条", flush=True)

        try:
            for i, (dep_state, dep_time) in enumerate(zip(departure_states, departure_times)):
                results = self._search_single_departure(
                    dep_state,
                    dep_time,
                    arrival_orbit,
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

    def _grid_search_parallel(
        self,
        departure_states: np.ndarray,
        departure_times: np.ndarray,
        arrival_orbit: Orbit,
        dep_name: str,
        arr_name: str,
        verbose: bool,
        n_workers: int,
    ) -> List[Dict[str, Any]]:
        """对多个出发点并行做与 `_grid_search_sequential` 相同的网格搜索。

        每个出发点调用 `_search_single_departure`（对 α 网格逐点求解转移）。
        使用 `ThreadPoolExecutor`：任务提交顺序为 0..N-1，但完成顺序由 `as_completed`
        决定；通过 `futures[future] -> i` 把每条结果标回原始出发点下标 `departure_time_index`。

        与串行版的差异：某出发点若抛错，仅跳过该点并（在 verbose 下）打印，不中断整批。
        """
        total_departures = len(departure_states)
        n_alpha = int(self.n_alpha)
        total_steps = total_departures * n_alpha
        all_results: List[Dict[str, Any]] = []

        worker_bars: Optional[List[Any]] = None
        aggregate_pbar: Optional[Any] = None
        aggregate_lock: Optional[threading.Lock] = None
        slot_queue: Optional["queue.Queue[int]"] = None

        if verbose and total_steps > 0:
            use_multiline = self._use_multiline_worker_tqdm(n_workers)
            if use_multiline:
                tqdm.write(
                    f"  并行搜索: {total_departures}×{n_alpha}={total_steps} 步 | {n_workers} 槽（分槽）"
                )
                worker_bars = self._open_parallel_worker_progress_bars(n_workers, n_alpha)
            else:
                tqdm.write(
                    f"  并行搜索: {total_departures}×{n_alpha}={total_steps} 步 | {n_workers} 槽（单行）"
                )
                aggregate_lock = threading.Lock()
                aggregate_pbar = self._open_search_progress_bar(total_steps, "并行网格搜索")
            slot_queue = queue.Queue()
            for slot in range(n_workers):
                slot_queue.put(slot)

        try:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                if worker_bars is not None and slot_queue is not None:
                    futures = {
                        executor.submit(
                            self._run_departure_with_worker_slot,
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
                            zip(departure_states, departure_times)
                        )
                    }
                elif aggregate_pbar is not None and slot_queue is not None:
                    futures = {
                        executor.submit(
                            self._run_departure_with_worker_slot,
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
                            zip(departure_states, departure_times)
                        )
                    }
                else:
                    futures = {
                        executor.submit(
                            self._search_single_departure,
                            dep_state,
                            dep_time,
                            arrival_orbit,
                            False,
                            None,
                            i,
                        ): i
                        for i, (dep_state, dep_time) in enumerate(
                            zip(departure_states, departure_times)
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

    def _search_single_departure(
        self,
        departure_state: np.ndarray,
        departure_time: float,
        arrival_orbit: Orbit,
        verbose: bool = False,
        pbar: Optional[Any] = None,
        departure_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """对单个出发点搜索α网格

        参数:
            verbose: 未传 ``pbar`` 时是否打印 α 文本进度。
            pbar: 每 α 步 ``update(1)``；可为分槽条或 ``_AggregatePbarWithSlot``。
            departure_index: 当前出发点下标，用于 postfix。
        """
        results = []

        alpha_grid = np.linspace(self.alpha_min, self.alpha_max, self.n_alpha)
        n_alpha = int(self.n_alpha)

        for i_alpha, alpha in enumerate(alpha_grid, start=1):
            try:
                if verbose:
                    pct = i_alpha / n_alpha * 100
                    print(f"    α 进度: {i_alpha}/{n_alpha} ({pct:.1f}%)", flush=True)
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
                    )  # //TODO 这个地方积分特别耗时
                except Exception:
                    result = {
                        "success": False,
                        "departure_state": departure_state,
                        "departure_time": departure_time,
                        "alpha": alpha,
                        "status": "integration_failed",
                    }
                    results.append(result)
                else:
                    collision, body, col_idx = self._check_collision(traj_states)
                    min_dist, min_idx = self._compute_min_distance(traj_states, arrival_orbit)
                    intersection, int_point, int_idx = self._detect_intersection(
                        traj_states, arrival_orbit, self.intersection_threshold
                    )
                    local_min, local_min_dist, local_min_idx = self._detect_local_minimum(
                        traj_states, arrival_orbit
                    )

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
                        "intersection_found": intersection,
                        "intersection_point": int_point,
                        "intersection_idx": int_idx,
                        "local_minimum_found": local_min,
                        "local_minimum_distance": local_min_dist,
                        "local_minimum_idx": local_min_idx,
                        "collision_found": collision,
                        "collision_body": body,
                        "collision_idx": col_idx,
                    }

                    if collision:
                        result["status"] = "collision"
                    elif intersection:
                        result["status"] = "success"
                    elif min_dist < self.min_distance_threshold:
                        result["status"] = "success"
                    else:
                        result["status"] = "no_intersection"

                    results.append(result)
            finally:
                if pbar is not None:
                    pbar.update(1)
                    if departure_index is not None:
                        pbar.set_postfix_str(f"dep={departure_index} α={alpha:.4f}")

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
        """在 CR3BP 下从 ``initial_state`` 前向积分到 ``transfer_time``，得到等间隔采样轨迹。

        使用 ``self.dynamics.propagate``（``solve_ivp`` + 稠密 ``t_eval``）。``with_stm=False``、
        ``with_jacobi=False`` 避免 STM 增广维与逐点 Jacobi，减轻搜索阶段开销。输出步数约为
        ``transfer_time / dt``，供后续碰撞检测、与目标轨道的距离/相交分析使用；积分总耗时主要取决于
        ``max_transfer_time``、``integration_dt`` 以及动力学对象上的 ``rtol`` / ``atol`` /
        ``max_step``，而非本函数本身。

        参数:
            initial_state: 六维状态 ``[x,y,z,vx,vy,vz]``（无量纲）。
            transfer_time: 积分时长上界（与 ``self.max_transfer_time`` 一致，无量纲）。
            dt: 输出时间步长 ``self.integration_dt``；过密会增大 ``t_eval`` 长度与插值开销。

        返回:
            ``(states, times)``，与 ``propagate`` 返回的 ``states`` / ``time`` 一致。
        """
        n_steps = max(int(transfer_time / dt) + 1, 2)
        t_eval = np.linspace(0, transfer_time, n_steps)

        result = self.dynamics.propagate(
            initial_state=initial_state,
            t_span=[0, transfer_time],
            t_eval=t_eval,
            with_stm=False,
            with_jacobi=False,
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
    dep_name: str,
    arr_name: str,
) -> List[Dict[str, Any]]:
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

    results = searcher._search_single_departure(dep_state, dep_time, arrival_orbit)
    for r in results:
        r["departure_orbit_name"] = dep_name
        r["arrival_orbit_name"] = arr_name
        r["departure_time_index"] = idx
    return results


def load_orbit_from_json(filepath: str) -> Orbit:
    """从JSON文件加载轨道数据

    参数:
        filepath: JSON文件路径

    返回:
        Orbit对象
    """
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
    results: List[Dict[str, Any]],
    filepath: str,
) -> None:
    """保存搜索结果到JSON文件

    参数:
        results: 搜索结果列表
        filepath: 输出文件路径
    """
    output = []
    for r in results:
        result_dict = {
            "departure_orbit_name": r.get("departure_orbit_name", ""),
            "arrival_orbit_name": r.get("arrival_orbit_name", ""),
            "departure_time_index": r.get("departure_time_index", -1),
            "departure_time": float(r.get("departure_time", 0.0)),
            "alpha": float(r.get("alpha", 0.0)),
            "transfer_time": float(r.get("transfer_time", 0.0)),
            "intersection_found": r.get("intersection_found", False),
            "min_distance": float(r.get("min_distance", float("inf"))),
            "local_minimum_found": r.get("local_minimum_found", False),
            "collision_found": r.get("collision_found", False),
            "status": r.get("status", "unknown"),
        }
        output.append(result_dict)

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
