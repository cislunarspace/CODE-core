"""
轨道转移搜索模块

实现论文Cui et al. (2025)中的"搜索-优化"两步法的搜索阶段。
专门用于平面转移轨道设计，搜索变量: 出发点位置、α(切向速度比)

使用方式:
    transfer = TransferSearch(system, dynamics)
    transfer.set_departure_orbit(departure_orbit).set_arrival_orbit(arrival_orbit)
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
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from tqdm.auto import tqdm


class _AggregatePbarWithSlot:
    """共享总进度：转发 update/postfix，加锁并加槽位前缀。"""

    __slots__ = ("_inner", "_lock", "_slot")

    def __init__(self, inner: Any, lock: threading.Lock | None, slot: int) -> None:
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


from ..core.dynamics import CR3BP_Dynamics  # noqa: E402
from ..core.orbit import Orbit  # noqa: E402
from ..core.cr3bp_system import CR3BP_System  # noqa: E402
from .search_config import SearchConfig  # noqa: E402
from .transfer_optimization import (  # noqa: E402
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
)
from .config import TransferOptimizationResult  # noqa: E402

# 100 km 换算为 CR3BP 无量纲单位：用于判断轨迹-轨道最近距离是否"足够近"
DEFAULT_MIN_DISTANCE_THRESHOLD_DU = 100.0 / CR3BP_System.EARTH_MOON_DISTANCE_KM


class TransferSearch:
    """通用轨道转移搜索算法

    实现论文Section III.A的搜索阶段算法:
    1. 从出发点轨道等时间间隔采样
    2. 对每个出发点，网格化搜索α
    3. 前向积分获取转移轨迹
    4. 筛选与目标轨道相交或距离局部最小的候选解

    搜索参数集中存储在 ``self.config``（:class:`SearchConfig` dataclass）中，
    同时通过属性代理提供向后兼容的直接读写访问（``self.alpha_min`` 等）。

    使用方式:
        transfer = TransferSearch(dynamics)
        transfer.configure_search(alpha_min=0.5, alpha_max=2.5, n_alpha=101)
        results = transfer.search(...)

    或通过 SearchConfig:
        from e2m2e.transfer import SearchConfig
        cfg = SearchConfig(alpha_min=0.5, alpha_max=2.5, n_alpha=101)
        transfer = TransferSearch(dynamics, config=cfg)
    """

    # --- 搜索配置属性名（与 SearchConfig 字段一一对应） ---
    _CONFIG_FIELDS: tuple[str, ...] = (
        "alpha_min",
        "alpha_max",
        "n_alpha",
        "n_departure",
        "max_transfer_time",
        "intersection_threshold",
        "min_distance_threshold",
        "collision_earth_radius",
        "collision_moon_radius",
        "integration_dt",
        "alpha_range",
        "transfer_time_range",
        "t_ins_range",
        "velocity_angle_tolerance",
    )

    def __init__(
        self,
        dynamics: CR3BP_Dynamics,
        name: str = "TransferSearch",
        config: SearchConfig | None = None,
    ):
        """初始化转移搜索器。

        Args:
            dynamics: CR3BP 动力学对象（需提供 ``system``、``propagate`` 等）。
            name: 搜索器实例名称，用于日志输出。
            config: 搜索/优化配置；为 ``None`` 时使用默认 ``SearchConfig()``。
        """
        self.system = dynamics.system
        # dynamics 类型说明：
        # - 内部搜索积分（_forward_integrate）仅调用 propagate()，满足 Propagator Protocol
        # - 但 __init__ 还访问 dynamics.system 与 system.mu，因此保留 CR3BP_Dynamics 具体类型
        self.dynamics: CR3BP_Dynamics = dynamics
        self.mu = self.system.mu
        self.name = name
        self._departure_orbit: Orbit | None = None
        self._arrival_orbit: Orbit | None = None
        self._search_results: list[dict[str, Any]] | None = None
        self._optimized_result: Any = None
        self._verbose = True
        self._n_workers: int | None = None
        # "processes" 多进程，利于 CPU 密集积分绕过 GIL；"threads" 保留线程内 tqdm 细粒度进度
        self._parallel_backend: str = "processes"

        # 搜索 + 优化配置（集中管理）
        self._config: SearchConfig = config if config is not None else SearchConfig()

    # --- 向后兼容属性代理：读/写直接转发到 _config ---

    @property
    def config(self) -> SearchConfig:
        """搜索/优化配置对象。"""
        return self._config

    @config.setter
    def config(self, value: SearchConfig) -> None:
        self._config = value

    def __getattr__(self, name: str) -> Any:
        """属性代理：读取 ``_CONFIG_FIELDS`` 中的字段时转发到 ``self._config``。"""
        # 仅代理 _CONFIG_FIELDS 中的字段，避免干扰其他属性查找
        if name in TransferSearch._CONFIG_FIELDS:
            return getattr(self._config, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """属性代理：写入 ``_CONFIG_FIELDS`` 中的字段时转发到 ``self._config``。"""
        if name in TransferSearch._CONFIG_FIELDS:
            setattr(self._config, name, value)
        else:
            super().__setattr__(name, value)

    def set_verbose(self, verbose: bool) -> TransferSearch:
        """设置是否输出详细信息"""
        self._verbose = verbose
        return self

    def set_n_workers(self, n_workers: int) -> TransferSearch:
        """设置并行worker数量"""
        self._n_workers = n_workers
        return self

    def set_parallel_backend(self, backend: str) -> TransferSearch:
        """设置并行后端：``processes``（默认，多进程）或 ``threads``（多线程）。"""
        b = backend.strip().lower()
        if b not in ("processes", "threads"):
            raise ValueError("parallel_backend 须为 'processes' 或 'threads'")
        self._parallel_backend = b
        return self

    def configure_search(self, **kwargs) -> TransferSearch:
        """配置搜索参数（向后兼容，推荐直接设置实例属性）"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def _is_feasible(self, result: dict[str, Any]) -> bool:
        """可行候选：无碰撞，且（相交 / 全局最小距离或局部极小距离小于
        ``min_distance_threshold``）。"""
        mdt = self.min_distance_threshold
        if mdt is None:
            mdt = DEFAULT_MIN_DISTANCE_THRESHOLD_DU
        if result.get("collision_found", False):
            return False
        md = float(result.get("min_distance", float("inf")))
        lmd = float(result.get("local_minimum_distance", float("inf")))
        if result.get("intersection_found", False):
            return True
        if md < mdt:
            return True
        return bool(result.get("local_minimum_found", False) and lmd < mdt)

    def get_feasible_results(self) -> list[dict[str, Any]]:
        """获取所有可行搜索结果"""
        if self._search_results is None:
            return []
        return [r for r in self._search_results if self._is_feasible(r)]

    def search(
        self,
        *,
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
        departure_orbit: Orbit | None = None,
        arrival_orbit: Orbit | None = None,
        verbose: bool = True,
        n_workers: int | None = None,
        parallel_backend: str = "processes",
    ) -> list[dict[str, Any]]:
        """执行网格搜索

        Args:
            alpha_min: α 下界
            alpha_max: α 上界
            n_alpha: α 方向网格点数
            n_departure: 出发点采样数量
            max_transfer_time: 最大转移时间（CR3BP 无量纲时间）
            intersection_threshold: 相交判定距离阈值
            min_distance_threshold: 候选解距离阈值
            collision_earth_radius: 地球碰撞检测半径
            collision_moon_radius: 月球碰撞检测半径
            integration_dt: 积分时间步长
            departure_orbit: 出发轨道（可选，未提供则使用已设置的轨道）
            arrival_orbit: 目标轨道（可选，未提供则使用已设置的轨道）
            verbose: 是否输出详细信息（含进度）
            n_workers: 并行 worker 数量
            parallel_backend: ``processes``（默认）或 ``threads``

        Returns:
            搜索结果列表
        """
        dep_orbit = departure_orbit if departure_orbit is not None else self._departure_orbit
        arr_orbit = arrival_orbit if arrival_orbit is not None else self._arrival_orbit

        if dep_orbit is None or arr_orbit is None:
            raise ValueError("必须提供 departure_orbit 和 arrival_orbit")

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

        self._departure_orbit = dep_orbit
        self._arrival_orbit = arr_orbit

        if verbose:
            print(f"\n{'=' * 60}")
            print("转移轨道网格搜索")
            print(f"{'=' * 60}")
            print(f"出发点: {dep_orbit}")
            print(f"目标: {arr_orbit}")
            print(f"α范围: [{alpha_min}, {alpha_max}], n={n_alpha}")
            print(f"出发点数量: {n_departure}")
            print(f"{'=' * 60}\n")

        results = self._grid_search(
            dep_orbit,
            arr_orbit,
            verbose,
            n_workers,
            parallel_backend,
        )

        self._search_results = results

        if verbose:
            feasible = [r for r in results if self._is_feasible(r)]
            print(f"\n{'=' * 60}")
            print("搜索完成")
            print(f"  总候选解: {len(results)}")
            print(f"  可行解: {len(feasible)}")
            print(f"{'=' * 60}")

        return results

    def optimize(self, initial_guess: dict[str, Any] | None = None) -> TransferOptimizationResult:
        """执行优化

        Args:
            initial_guess: 优化初始猜测（通常来自搜索结果）

        Returns:
            优化结果
        """
        if self._departure_orbit is None or self._arrival_orbit is None:
            raise ValueError("必须先设置departure_orbit和arrival_orbit")

        if initial_guess is None:
            if self._search_results:
                feasible = self.get_feasible_results()
                initial_guess = feasible[0] if feasible else self._search_results[0]
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
        n_workers: int | None,
        parallel_backend: str,
    ) -> list[dict[str, Any]]:
        """网格搜索分发：根据 worker 数和后端选择串行/多进程/多线程。

        Args:
            departure_orbit: 出发轨道。
            arrival_orbit: 目标轨道。
            verbose: 是否输出进度信息。
            n_workers: 并行 worker 数；``None`` 时使用 CPU 核心数。
            parallel_backend: ``"processes"`` 或 ``"threads"``。

        Returns:
            搜索结果列表。
        """
        dep_name = getattr(departure_orbit, "name", "unknown")
        arr_name = getattr(arrival_orbit, "name", "unknown")

        departure_states, departure_times = self._sample_departure_points(departure_orbit)

        if n_workers is None:
            n_workers = multiprocessing.cpu_count()

        pb = parallel_backend.strip().lower()
        if pb not in ("processes", "threads"):
            raise ValueError("parallel_backend 须为 'processes' 或 'threads'")

        if n_workers == 1:
            return self._grid_search_sequential(
                departure_states,
                departure_times,
                arrival_orbit,
                dep_name,
                arr_name,
                verbose,
            )
        elif pb == "processes":
            return self._grid_search_parallel_processes(
                departure_states,
                departure_times,
                arrival_orbit,
                dep_name,
                arr_name,
                verbose,
                n_workers,
            )
        else:
            return self._grid_search_parallel_threads(
                departure_states,
                departure_times,
                arrival_orbit,
                dep_name,
                arr_name,
                verbose,
                n_workers,
            )

    def _sample_departure_points(self, departure_orbit: Orbit) -> tuple[np.ndarray, np.ndarray]:
        """从星历中均匀下采样出发点：直接使用 ``Orbit.states`` / ``Orbit.times`` 已有行，不积分。

        要求 ``n_departure <= len(times)``，否则报错。
        """
        times = departure_orbit.times
        states = departure_orbit.states
        n_pts = len(times)
        if n_pts == 0:
            raise ValueError("出发轨道无数据点")
        n_dep = self.n_departure
        if n_dep is None:
            raise ValueError("n_departure 未设置")
        n = int(n_dep)
        if n <= 0:
            raise ValueError("n_departure 须为正整数")
        if n > n_pts:
            raise ValueError(
                f"n_departure（{n}）不能大于出发轨道星历点数（{n_pts}），"
                f"请减小 n_departure 或增加星历密度"
            )

        n_sample = n
        if n_sample == 1:
            idx = np.array([0], dtype=int)
        else:
            idx = (
                (np.arange(n_sample, dtype=float) * (n_pts - 1) / (n_sample - 1))
                .round()
                .astype(int)
            )

        return states[idx].copy(), times[idx].copy()

    def _open_search_progress_bar(self, total: int, desc: str) -> Any | None:
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
        """非 TTY 用单行总进度；否则可分槽多行。
        环境变量 ``E2M2E_TQDM_MULTILINE`` 可强制开/关；槽数>32 只用单行。"""
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

    def _open_parallel_worker_progress_bars(self, n_workers: int, n_alpha: int) -> list[Any]:
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
    ) -> list[dict[str, Any]]:
        """串行网格搜索：逐出发点、逐α执行搜索。

        Args:
            departure_states: 出发点状态数组，形状 ``(n, 6)``。
            departure_times: 出发点时间数组。
            arrival_orbit: 目标轨道。
            dep_name: 出发轨道名称（写入结果字典）。
            arr_name: 目标轨道名称。
            verbose: 是否显示进度条。

        Returns:
            所有出发点的搜索结果列表。
        """
        all_results = []
        total_departures = len(departure_states)
        n_alpha_v = self.n_alpha
        if n_alpha_v is None:
            raise ValueError("n_alpha 未设置")
        n_alpha = int(n_alpha_v)
        total_steps = total_departures * n_alpha
        pbar = None
        if verbose and total_steps > 0:
            pbar = self._open_search_progress_bar(total_steps, "网格搜索")

        if verbose and total_steps <= 0:
            print(f"  总迭代步数: {total_steps}（出发点 × α），无进度条", flush=True)

        try:
            for i, (dep_state, dep_time) in enumerate(
                zip(departure_states, departure_times, strict=False)
            ):
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

    def _grid_search_parallel_processes(
        self,
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
        n_alpha_v = self.n_alpha
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
                f"  并行搜索(进程): {total_departures}×{n_alpha}"
                f"={total_steps} 步 | {n_workers} 进程"
            )
            pbar = self._open_search_progress_bar(total_steps, "并行网格搜索(进程)")
            # 多进程下主进程无法把 tqdm 传入子进程；子进程每 α 步向队列 put，主进程消费更新总条。
            # 须用 Manager().Queue()：原生 Value/Queue 不能经 ProcessPoolExecutor 参数传入 worker。
            progress_manager = multiprocessing.Manager()
            progress_queue = progress_manager.Queue()
            poll_stop = threading.Event()

            def _poll_process_progress() -> None:
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

            poll_thread = threading.Thread(target=_poll_process_progress, daemon=True)
            poll_thread.start()

        dyn = self.dynamics
        a0 = self.alpha_min
        a1 = self.alpha_max
        na = self.n_alpha
        nd = self.n_departure
        mtt = self.max_transfer_time
        ith = self.intersection_threshold
        mdt = self.min_distance_threshold
        cer = self.collision_earth_radius
        cmr = self.collision_moon_radius
        idt = self.integration_dt
        if (
            a0 is None
            or a1 is None
            or na is None
            or nd is None
            or mtt is None
            or ith is None
            or mdt is None
            or cer is None
            or cmr is None
            or idt is None
        ):
            raise ValueError(
                "请先设置 alpha_min, alpha_max, n_alpha, n_departure, max_transfer_time, "
                "intersection_threshold, min_distance_threshold, collision_earth_radius, "
                "collision_moon_radius, integration_dt"
            )
        pack_base = (
            self.mu,
            float(a0),
            float(a1),
            int(na),
            int(nd),
            float(mtt),
            float(ith),
            float(mdt),
            float(cer),
            float(cmr),
            float(idt),
            str(dyn.integrator),
            float(dyn.rtol),
            float(dyn.atol),
            float(dyn.max_step),
            dep_name,
            arr_name,
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
                    fut = executor.submit(_process_departure_worker_packed, packed)
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

    def _grid_search_parallel_threads(
        self,
        departure_states: np.ndarray,
        departure_times: np.ndarray,
        arrival_orbit: Orbit,
        dep_name: str,
        arr_name: str,
        verbose: bool,
        n_workers: int,
    ) -> list[dict[str, Any]]:
        """多线程并行搜索：支持细粒度 tqdm 进度条。

        受 GIL 限制，CPU 利用率常低于多进程；适用于 I/O 或需实时进度的场景。
        每个出发点调用 ``_search_single_departure``，抛错时跳过。

        Args:
            departure_states: 出发点状态数组。
            departure_times: 出发点时间数组。
            arrival_orbit: 目标轨道。
            dep_name: 出发轨道名称。
            arr_name: 目标轨道名称。
            verbose: 是否显示进度条。
            n_workers: 线程数。

        Returns:
            搜索结果列表。
        """
        total_departures = len(departure_states)
        n_alpha_v = self.n_alpha
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
            use_multiline = self._use_multiline_worker_tqdm(n_workers)
            if use_multiline:
                tqdm.write(
                    f"  并行搜索: {total_departures}×{n_alpha}"
                    f"={total_steps} 步 | {n_workers} 槽（分槽）"
                )
                worker_bars = self._open_parallel_worker_progress_bars(n_workers, n_alpha)
            else:
                tqdm.write(
                    f"  并行搜索: {total_departures}×{n_alpha}"
                    f"={total_steps} 步 | {n_workers} 槽（单行）"
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
                            zip(departure_states, departure_times, strict=False)
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
                            zip(departure_states, departure_times, strict=False)
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

    def _search_single_departure(
        self,
        departure_state: np.ndarray,
        departure_time: float,
        arrival_orbit: Orbit,
        verbose: bool = False,
        pbar: Any | None = None,
        departure_index: int | None = None,
        progress_queue: Any | None = None,
    ) -> list[dict[str, Any]]:
        """对单个出发点搜索 α 网格。

        对每个 α 值计算出发速度、前向积分，然后检测碰撞/相交/最近距离。

        Args:
            departure_state: 出发点六维状态 ``[x,y,z,vx,vy,vz]``。
            departure_time: 出发点时间。
            arrival_orbit: 目标轨道。
            verbose: 未传 ``pbar`` 时是否打印 α 文本进度。
            pbar: 每 α 步 ``update(1)``；可为分槽条或 ``_AggregatePbarWithSlot``。
            departure_index: 当前出发点下标，用于 postfix。
            progress_queue: 可选 ``Manager().Queue()`` 代理；每完成一 α 步 ``put(1)``，
                供主进程消费并更新总 tqdm。

        Returns:
            该出发点下所有 α 对应的搜索结果字典列表。
        """
        results = []

        a0 = self.alpha_min
        a1 = self.alpha_max
        na = self.n_alpha
        mtt = self.max_transfer_time
        idt = self.integration_dt
        ith = self.intersection_threshold
        mdt = self.min_distance_threshold
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

        for i_alpha, alpha in enumerate(alpha_grid, start=1):
            try:
                if verbose:
                    pct = i_alpha / n_alpha * 100
                    print(f"    α 进度: {i_alpha}/{n_alpha} ({pct:.1f}%)", flush=True)
                new_vel = self._compute_departure_velocity(departure_state, alpha)
                dv_departure = float(np.linalg.norm(new_vel - departure_state[3:6]))

                initial_state = np.concatenate(
                    [
                        departure_state[:3],
                        new_vel,
                    ]
                )

                try:
                    traj_states, traj_times = self._forward_integrate(initial_state, mtt, idt)
                except Exception:
                    result = {
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
                    results.append(result)
                else:
                    collision, body, col_idx = self._check_collision(traj_states)
                    # 一次性算出每步距离序列，下面所有"最近点/相交/首次穿越"判定共用同一份数据
                    d_per_step, orbit_idx_per_step = self._compute_distance_series(
                        traj_states, arrival_orbit
                    )
                    min_idx = int(np.argmin(d_per_step))
                    min_dist = float(d_per_step[min_idx])
                    orbit_idx = int(orbit_idx_per_step[min_idx])
                    v_tr = traj_states[min_idx][3:6]
                    v_ro = arrival_orbit.states[orbit_idx][3:6]
                    dv_insertion = float(np.linalg.norm(v_tr - v_ro))  # 粗估：最近几何点处速度差
                    intersection, int_point, int_idx = self._detect_intersection(
                        traj_states, arrival_orbit, ith
                    )
                    local_min, local_min_dist, local_min_idx = self._detect_local_minimum(
                        traj_states, arrival_orbit
                    )

                    # 首次进入两类阈值内的索引/时间（C1: 两对独立字段）。
                    # None 表示从未进入；保留 D1 语义（即使 idx=0 也忠实记录，绘图端做 fallback）。
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

    def _compute_departure_velocity(self, orbit_state: np.ndarray, alpha: float) -> np.ndarray:
        """计算出发点速度扰动 (平面)"""
        pos = orbit_state[:3]
        vel = orbit_state[3:]

        r_xy = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
        if r_xy < 1e-10:
            warnings.warn("位置靠近原点，使用原始速度", stacklevel=2)
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """在 CR3BP 下从 ``initial_state`` 前向积分到 ``transfer_time``，得到等间隔采样轨迹。

        使用 ``self.dynamics.propagate``（``solve_ivp`` + 稠密 ``t_eval``）。``with_stm=False``、
        ``with_jacobi=False`` 避免 STM 增广维与逐点 Jacobi，减轻搜索阶段开销。输出步数约为
        ``transfer_time / dt``，供后续碰撞检测、与目标轨道的距离/相交分析使用；积分总耗时主要取决于
        ``max_transfer_time``、``integration_dt`` 以及动力学对象上的 ``rtol`` / ``atol`` /
        ``max_step``，而非本函数本身。

        Args:
            initial_state: 六维状态 ``[x,y,z,vx,vy,vz]``（无量纲）。
            transfer_time: 积分时长上界（与 ``self.max_transfer_time`` 一致，无量纲）。
            dt: 输出时间步长 ``self.integration_dt``；过密会增大 ``t_eval`` 长度与插值开销。

        Returns:
            ``(states, times)``，与 ``propagate`` 返回的 ``states`` / ``time`` 一致。
        """
        n_steps = max(int(transfer_time / dt) + 1, 2)
        t_eval = np.linspace(0, transfer_time, n_steps)

        result = self.dynamics.propagate(
            initial_state=initial_state,
            t_span=(0.0, transfer_time),
            t_eval=t_eval,
            with_stm=False,
            with_jacobi=False,
        )

        return result["states"], result["time"]

    def _compute_distance_series(
        self, trajectory_states: np.ndarray, arrival_orbit: Orbit
    ) -> tuple[np.ndarray, np.ndarray]:
        """逐步计算轨迹各点到目标轨道的最小距离。

        返回与轨迹同长度的两个一维数组:
            d_per_step[i]          : 轨迹第 i 步到 arrival_orbit 最近点的距离
            orbit_idx_per_step[i]  : 该最近点在 arrival_orbit 上的采样下标

        ``_compute_min_distance``、``search()`` 的首次穿越扫描共用同一份距离序列，
        从而保证 ``min_distance``、``intersection_found``、``first_*_idx``
        三类衍生量在数值上严格自洽。
        """
        traj_positions = trajectory_states[:, :3]
        orbit_positions = arrival_orbit.states[:, :3]

        n_traj = len(traj_positions)
        n_orbit = len(orbit_positions)
        # 10_000_000 对 × 3坐标 × 8字节 ≈ 240 MB float64；
        # 超过此阈值时分块计算，避免内存溢出
        max_pairs = 10_000_000

        if n_traj * n_orbit > max_pairs:
            return self._compute_distance_series_chunked(traj_positions, orbit_positions)

        diff = traj_positions[:, np.newaxis, :] - orbit_positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))  # (n_traj, n_orbit)

        orbit_idx_per_step = np.argmin(distances, axis=1)
        d_per_step = distances[np.arange(n_traj), orbit_idx_per_step]

        return d_per_step, orbit_idx_per_step.astype(np.int64)

    def _compute_distance_series_chunked(
        self, traj_positions: np.ndarray, orbit_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """分块计算每步最近距离序列，避免大轨迹/大轨道时内存溢出。"""
        n_traj = len(traj_positions)
        n_orbit = len(orbit_positions)
        chunk_size = max(1, 10_000_000 // n_orbit)

        d_per_step = np.empty(n_traj, dtype=np.float64)
        orbit_idx_per_step = np.empty(n_traj, dtype=np.int64)

        for start in range(0, n_traj, chunk_size):
            end = min(start + chunk_size, n_traj)
            chunk = traj_positions[start:end]

            diff = chunk[:, np.newaxis, :] - orbit_positions[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))  # (chunk_len, n_orbit)

            chunk_orbit_idx = np.argmin(distances, axis=1)
            chunk_d = distances[np.arange(end - start), chunk_orbit_idx]

            d_per_step[start:end] = chunk_d
            orbit_idx_per_step[start:end] = chunk_orbit_idx

        return d_per_step, orbit_idx_per_step

    def _compute_min_distance(
        self, trajectory_states: np.ndarray, arrival_orbit: Orbit
    ) -> tuple[float, int, int]:
        """计算轨迹到目标轨道的最小距离及最近点（轨迹步、目标轨道采样下标）。

        薄包装：复用 :meth:`_compute_distance_series` 的完整距离序列，
        保留旧的 3 元组签名，所有现有调用方零改动。
        """
        d_per_step, orbit_idx_per_step = self._compute_distance_series(
            trajectory_states, arrival_orbit
        )
        step_idx = int(np.argmin(d_per_step))
        return float(d_per_step[step_idx]), step_idx, int(orbit_idx_per_step[step_idx])

    def compute_min_distance_to_orbit(
        self, trajectory_states: np.ndarray, arrival_orbit: Orbit
    ) -> tuple[float, int]:
        """轨迹到目标轨道的最小距离及对应轨迹步索引（公开 API）。"""
        md, si, _ = self._compute_min_distance(trajectory_states, arrival_orbit)
        return md, si

    def _detect_intersection(
        self,
        trajectory_states: np.ndarray,
        arrival_orbit: Orbit,
        threshold: float,
    ) -> tuple[bool, np.ndarray | None, int]:
        """检测轨迹是否与目标轨道相交"""
        min_dist, step_idx, _ = self._compute_min_distance(trajectory_states, arrival_orbit)

        if min_dist < threshold:
            return True, trajectory_states[step_idx], step_idx

        return False, None, -1

    def find_intersection(
        self,
        trajectory_states: np.ndarray,
        arrival_orbit: Orbit,
        threshold: float | None = None,
    ) -> tuple[bool, np.ndarray, int]:
        """检测轨迹是否与目标轨道相交（与 ``_detect_intersection`` 相同，
        默认使用 ``intersection_threshold``）。"""
        ith = threshold if threshold is not None else self.intersection_threshold
        if ith is None:
            raise ValueError("intersection_threshold 未设置且未传入 threshold")
        found, pt, idx = self._detect_intersection(trajectory_states, arrival_orbit, float(ith))
        if pt is None:
            return False, np.zeros(6), idx
        return found, pt, idx

    def _detect_local_minimum(
        self,
        trajectory_states: np.ndarray,
        arrival_orbit: Orbit,
    ) -> tuple[bool, float, int]:
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
    ) -> tuple[bool, str | None, int]:
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


DROTransferSearch = TransferSearch
DROROTransferSearch = TransferSearch


def _process_departure_worker(
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
        dep_state,
        dep_time,
        arrival_orbit,
        progress_queue=progress_queue,
    )
    for r in results:
        r["departure_orbit_name"] = dep_name
        r["arrival_orbit_name"] = arr_name
        r["departure_time_index"] = idx
    return results


def _process_departure_worker_packed(packed: tuple[Any, ...]) -> list[dict[str, Any]]:
    """单元组打包，供 ``ProcessPoolExecutor`` 提交。"""
    return _process_departure_worker(*packed)


def load_orbit_from_json(filepath: str) -> Orbit:
    """从 JSON 文件加载轨道数据

    Args:
        filepath: JSON 文件路径

    Returns:
        Orbit 对象
    """
    with open(filepath) as f:
        data = json.load(f)

    states = np.array(data["states"])
    times = np.array(data["times"])

    orbit = Orbit(states=states, times=times)

    if "orbit_type" in data:
        orbit.metadata["orbit_type"] = data["orbit_type"]
    if "period_ratio" in data:
        orbit.metadata["period_ratio"] = data["period_ratio"]

    return orbit
