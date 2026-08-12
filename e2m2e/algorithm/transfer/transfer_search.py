"""轨道转移搜索编排类。

实现论文 Cui et al. (2025) 中"搜索-优化"两步法的搜索阶段：
平面转移轨道设计，搜索变量: 出发点位置、α（切向速度比）。

本文件只负责高层编排：参数管理、``search()`` / ``optimize()`` 入口、
可行性过滤、对外公共 API。实际的

- 几何核（碰撞 / 距离 / 相交 / 局部极小）    → :mod:`search_geometry`
- tqdm 进度条封装                            → :mod:`search_progress`
- 进程 / 线程后端 + per-α 积分内核 + worker → :mod:`search_parallel`

为保持向后兼容（测试 ``monkeypatch.setattr(TransferSearch, "_forward_integrate", ...)``、
``searcher._private_method(...)`` 等调用），``TransferSearch`` 仍以 thin-wrapper
方式重新暴露原私有方法名。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ...data.types.orbit import Orbit
from ..dynamics import CR3BP_Dynamics, CR3BP_System
from ..results import TransferCandidateResult
from . import search_geometry, search_parallel
from .config import TransferConfig, TransferOptimizationResult
from .propulsion import ImpulsivePropulsion
from .transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
)

# 100 km 换算为 CR3BP 无量纲单位。
DEFAULT_MIN_DISTANCE_THRESHOLD_DU = 100.0 / CR3BP_System.EARTH_MOON_DISTANCE_KM


def _default_parallel_backend() -> str:
    """默认使用 Rust 后端；无扩展时在使用处报告扩展不可用。

    ``processes`` 和 ``threads`` 仅在调用方显式选择时保留。不能因 Rust 扩展
    缺失而悄然改变默认算法与并行模型（issue #378）。
    """
    return "rust"


class TransferSearch:
    """通用轨道转移搜索算法。

    1. 从出发点轨道等时间间隔采样
    2. 对每个出发点，网格化搜索 α
    3. 前向积分获取转移轨迹
    4. 筛选与目标轨道相交或距离局部最小的候选解

    搜索参数集中在 ``self.config``（:class:`TransferConfig`）的
    ``search_*`` 字段中，``self.alpha_min`` 等裸名通过属性代理映射到前缀字段。
    """

    _CONFIG_FIELD_MAP: dict[str, str] = {
        "alpha_min": "search_alpha_min",
        "alpha_max": "search_alpha_max",
        "n_alpha": "search_n_alpha",
        "n_departure": "search_n_departure",
        "max_transfer_time": "search_max_transfer_time",
        "intersection_threshold": "search_intersection_threshold",
        "min_distance_threshold": "search_min_distance_threshold",
        "collision_earth_radius": "search_collision_earth_radius",
        "collision_moon_radius": "search_collision_moon_radius",
        "integration_dt": "search_integration_dt",
        "alpha_range": "nlp_alpha_range",
        "transfer_time_range": "nlp_transfer_time_range",
        "t_ins_range": "nlp_t_ins_range",
    }

    def __init__(
        self,
        dynamics: CR3BP_Dynamics,
        name: str = "TransferSearch",
        config: TransferConfig | None = None,
    ):
        self.system = dynamics.system
        self.dynamics: CR3BP_Dynamics = dynamics
        self.mu = self.system.mu
        self.name = name
        self._departure_orbit: Orbit | None = None
        self._arrival_orbit: Orbit | None = None
        self._search_results: list[TransferCandidateResult] | None = None
        self._optimized_result: Any = None
        self._verbose = True
        self._n_workers: int | None = None
        self._parallel_backend: str = _default_parallel_backend()
        self._propulsion = ImpulsivePropulsion()
        self._config: TransferConfig = config if config is not None else TransferConfig()

    @property
    def config(self) -> TransferConfig:
        return self._config

    @config.setter
    def config(self, value: TransferConfig) -> None:
        self._config = value

    def __getattr__(self, name: str) -> Any:
        """属性代理：读取映射中的裸名时转发到 ``self._config``。"""
        field = TransferSearch._CONFIG_FIELD_MAP.get(name)
        if field is not None:
            return getattr(self._config, field)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """属性代理：写入映射中的裸名时转发到 ``self._config``。"""
        field = TransferSearch._CONFIG_FIELD_MAP.get(name)
        if field is not None:
            setattr(self._config, field, value)
        else:
            super().__setattr__(name, value)

    def set_verbose(self, verbose: bool) -> TransferSearch:
        self._verbose = verbose
        return self

    def set_n_workers(self, n_workers: int) -> TransferSearch:
        self._n_workers = n_workers
        return self

    def set_parallel_backend(self, backend: str) -> TransferSearch:
        """设置并行后端：``rust``（默认）、``processes`` 或 ``threads``。

        默认恒为 ``rust``（:func:`_default_parallel_backend` 固定返回）：
        Rust 扩展缺失时在使用处直接报错（issue #378，不静默回退
        ``processes``）。``rust`` 走 Rust+Rayon 内核；几何方法被 monkeypatch
        （测试注入缝）时回退 Python 路径，生产路径不触发。
        ``processes``/``threads`` 行为不变。
        """
        b = backend.strip().lower()
        if b not in ("processes", "threads", "rust"):
            raise ValueError("parallel_backend 须为 'processes'、'threads' 或 'rust'")
        self._parallel_backend = b
        return self

    def configure_search(self, **kwargs) -> TransferSearch:
        """配置搜索参数。"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def get_feasible_results(self) -> list[TransferCandidateResult]:
        if self._search_results is None:
            return []
        return [r for r in self._search_results if self._is_feasible(r)]

    def search(
        self,
        config: TransferConfig | None = None,
        *,
        alpha_min: float | None = None,
        alpha_max: float | None = None,
        n_alpha: int | None = None,
        n_departure: int | None = None,
        max_transfer_time: float | None = None,
        intersection_threshold: float | None = None,
        min_distance_threshold: float | None = None,
        collision_earth_radius: float | None = None,
        collision_moon_radius: float | None = None,
        integration_dt: float | None = None,
        departure_orbit: Orbit | None = None,
        arrival_orbit: Orbit | None = None,
        verbose: bool = True,
        n_workers: int | None = None,
        parallel_backend: str | None = None,
    ) -> list[TransferCandidateResult]:
        """执行网格搜索。"""
        # None → 用实例属性（由 set_parallel_backend / _default_parallel_backend 设）；
        # 显式传入则覆盖。连起来后 set_parallel_backend 才对 search() 生效。
        parallel_backend = (
            parallel_backend if parallel_backend is not None else self._parallel_backend
        )
        dep_orbit = departure_orbit if departure_orbit is not None else self._departure_orbit
        arr_orbit = arrival_orbit if arrival_orbit is not None else self._arrival_orbit
        if dep_orbit is None or arr_orbit is None:
            raise ValueError("必须提供 departure_orbit 和 arrival_orbit")
        if config is not None:
            self._config = config

        for name, value in (
            ("alpha_min", alpha_min),
            ("alpha_max", alpha_max),
            ("n_alpha", n_alpha),
            ("n_departure", n_departure),
            ("max_transfer_time", max_transfer_time),
            ("intersection_threshold", intersection_threshold),
            ("min_distance_threshold", min_distance_threshold),
            ("collision_earth_radius", collision_earth_radius),
            ("collision_moon_radius", collision_moon_radius),
            ("integration_dt", integration_dt),
        ):
            if value is not None:
                setattr(self, name, value)

        self._departure_orbit = dep_orbit
        self._arrival_orbit = arr_orbit

        if verbose:
            print(f"\n{'=' * 60}")
            print("转移轨道网格搜索")
            print(f"{'=' * 60}")
            print(f"出发点: {dep_orbit}")
            print(f"目标: {arr_orbit}")
            print(f"α范围: [{self.alpha_min}, {self.alpha_max}], n={self.n_alpha}")
            print(f"出发点数量: {self.n_departure}")
            print(f"{'=' * 60}\n")

        results = search_parallel.dispatch_grid_search(
            self,
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

    def optimize(
        self, initial_guess: TransferCandidateResult | dict[str, Any] | None = None
    ) -> TransferOptimizationResult:
        """执行优化。"""
        if self._departure_orbit is None or self._arrival_orbit is None:
            raise ValueError("必须先设置 departure_orbit 和 arrival_orbit")
        if initial_guess is None:
            if self._search_results:
                feasible = self.get_feasible_results()
                initial_guess = feasible[0] if feasible else self._search_results[0]
            else:
                raise ValueError("必须提供 initial_guess 或先运行 search()")

        optimizer = DROTRONLPOptimizer(
            system=self.system,
            dynamics=self.dynamics,
            departure_orbit=self._departure_orbit,
            arrival_orbit=self._arrival_orbit,
            departure_state=initial_guess["departure_state"],
            propulsion=self._propulsion,
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

    def compute_min_distance_to_orbit(
        self, trajectory_states: np.ndarray, arrival_orbit: Orbit
    ) -> tuple[float, int]:
        """轨迹到目标轨道的最小距离及对应轨迹步索引（公开 API）。"""
        md, si, _ = search_geometry.compute_min_distance(trajectory_states, arrival_orbit)
        return md, si

    def find_intersection(
        self,
        trajectory_states: np.ndarray,
        arrival_orbit: Orbit,
        threshold: float | None = None,
    ) -> tuple[bool, np.ndarray, int]:
        """检测轨迹是否与目标轨道相交，默认使用 ``intersection_threshold``。"""
        ith = threshold if threshold is not None else self.intersection_threshold
        if ith is None:
            raise ValueError("intersection_threshold 未设置且未传入 threshold")
        found, pt, idx = search_geometry.detect_intersection(
            trajectory_states, arrival_orbit, float(ith)
        )
        if pt is None:
            return False, np.zeros(6), idx
        return found, pt, idx

    # === 私有方法向后兼容 thin-wrapper ===
    # 既有测试调用 ``searcher._x`` 或 ``monkeypatch.setattr(TransferSearch, "_x", ...)``；
    # 委托到 :mod:`search_geometry` / :mod:`search_parallel` 使这些调用在拆分后继续生效。

    def _is_feasible(self, result: TransferCandidateResult) -> bool:
        return search_geometry.is_feasible_result(
            result,
            self.min_distance_threshold,
            DEFAULT_MIN_DISTANCE_THRESHOLD_DU,
        )

    def _compute_distance_series(self, trajectory_states, arrival_orbit):
        return search_geometry.compute_distance_series(trajectory_states, arrival_orbit)

    def _compute_distance_series_chunked(self, traj_positions, orbit_positions):
        return search_geometry.compute_distance_series_chunked(traj_positions, orbit_positions)

    def _compute_min_distance(self, trajectory_states, arrival_orbit):
        return search_geometry.compute_min_distance(trajectory_states, arrival_orbit)

    def _detect_intersection(self, trajectory_states, arrival_orbit, threshold):
        return search_geometry.detect_intersection(trajectory_states, arrival_orbit, threshold)

    def _detect_local_minimum(self, trajectory_states, arrival_orbit):
        return search_geometry.detect_local_minimum(trajectory_states, arrival_orbit)

    def _check_collision(self, trajectory_states):
        return search_geometry.check_collision(
            trajectory_states,
            self.mu,
            self.collision_earth_radius,
            self.collision_moon_radius,
        )

    def _compute_departure_velocity(self, orbit_state, alpha):
        return self._propulsion.compute_departure_velocity(orbit_state, alpha=alpha)

    def _forward_integrate(self, initial_state, transfer_time, dt):
        return search_parallel.forward_integrate(self.dynamics, initial_state, transfer_time, dt)

    def _sample_departure_points(self, departure_orbit):
        return search_parallel.sample_departure_points(departure_orbit, self.n_departure)

    def _search_single_departure(
        self,
        departure_state,
        departure_time,
        arrival_orbit,
        verbose=False,
        pbar=None,
        departure_index=None,
        progress_queue=None,
    ):
        return search_parallel.search_single_departure(
            self,
            departure_state=departure_state,
            departure_time=departure_time,
            arrival_orbit=arrival_orbit,
            verbose=verbose,
            pbar=pbar,
            departure_index=departure_index,
            progress_queue=progress_queue,
        )


def load_orbit_from_json(filepath: str) -> Orbit:
    """从 JSON 文件加载轨道数据。"""
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
