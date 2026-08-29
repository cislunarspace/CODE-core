"""WSB 太阳引力辅助间接转移：BCR4BP 弹道搜索 + 到达段精化。

弱稳定边界（Weak Stability Boundary）转移利用地月 BCR4BP 动力学中的
太阳引力摄动，在近月点附近使航天器的相对月球 Kepler 能量 H₂ < 0
（弹道捕获判据——无需制动脉冲即被月球束缚），自然被月球捕获后可由
小量圆化脉冲稳定。总 Δv 仅来自出发脉冲和到达脉冲。

搜索空间：sun_phase × departure_phase × tof 三维网格。默认 Rust 后端把
候选参数化、BCR4BP 传播、截面检测和筛选交给 Rayon；Python 实现只在调用方
显式指定 ``backend="python"`` 时作为等价性参照，绝不自动回退。

BCR4BP 旋转系→惯性系速度修正：

    ``v_rel_moon = (vx - y, vy + x - (1-μ), vz)``

其中 (1-μ, 0, 0) 为月球在旋转系中的位置，减去月球惯性速度
``ω × r_moon = (0, 1-μ, 0)`` 得到相对月球的惯性系速度。
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ...data.constants import SECONDS_PER_DAY
from ...data.constants.bodies import MOON
from ...data.templates import ConvergenceState, FailureCause
from ...exceptions import PropagationFailure
from ..dynamics import BCR4BP_Dynamics, BCR4BPSystem, CR3BP_Dynamics, CR3BP_System
from ..manifold.sections import PoincareSection, detect_crossings
from ..results import CandidateSearchResult, ResultStatus

logger = logging.getLogger(__name__)

# 月球半径 (km)
R_MOON_KM: float = MOON.require_mean_radius_km()


@dataclass(frozen=True)
class WsbSearchParams:
    """WSB 弹道搜索参数。

    搜索空间：太阳相位角 × 出发相位角 × 飞行时间（TOF）。
    近月点高度由传播自然决定，不作为独立搜索变量。
    弹道捕获由 H₂ < 0 判定（Belbruno & Miller 1993）。

    Attributes:
        sun_phase_range: 太阳相位角范围 (min, max)，弧度，[0, 2pi)
        n_sun_phase: 太阳相位角网格点数
        departure_phase_range: 出发相位角范围 (min, max)，弧度，[0, 2pi)，
            即出发点在停泊轨道上的滑行角（绕地球旋转，改变月地几何）
        n_departure_phase: 出发相位角网格点数
        tof_range: 飞行时间范围 (min, max)，天（WSB 典型 90-150 天）
        n_tof: TOF 网格点数
        tli_speed_factor: TLI 脉冲速度与逃逸速度之比。典型值 < 1
            （如 0.99，远地点略超月球轨道的奔月轨道）；= 1 为抛物线逃逸
        rtol: 传播相对容差（网格筛选级，粗干精化；精化由 ThreeBodyLambert 打靶负责）
        atol: 传播绝对容差（同上）
        max_steps: 筛选阶段单条轨迹最大积分步数（仅 Rust 后端生效）。超过
            即判为传播失败丢弃；真候选典型几百步，深混沌擦月轨迹需几十万步
            且无筛选价值。Python 参照后端不截断，保持精确语义
        n_propagation_samples: 传播采样点数
        perilune_alt_min: 近月点高度下限 (km)
        perilune_alt_max: 近月点高度上限 (km)
        max_total_dv: 最大总 Δv 筛选阈值 (km/s)
        h2_energy_threshold: H₂ 能量阈值（无量纲），H₂ < 此值的候选保留（弹道捕获）
        n_propagation_samples: 传播采样点数
    """

    sun_phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)
    n_sun_phase: int = 50
    departure_phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)
    n_departure_phase: int = 180
    tof_range: tuple[float, float] = (90.0, 150.0)
    n_tof: int = 50
    tli_speed_factor: float = 0.99
    rtol: float = 1e-9
    atol: float = 1e-9
    max_steps: int = 20_000
    perilune_alt_min: float = 100.0
    perilune_alt_max: float = 10000.0
    max_total_dv: float = 5.0
    h2_energy_threshold: float = 0.0
    n_propagation_samples: int = 500

    def __post_init__(self) -> None:
        """验证参数范围。"""
        for name, range_val in [
            ("sun_phase_range", self.sun_phase_range),
            ("departure_phase_range", self.departure_phase_range),
        ]:
            lo, hi = range_val
            if lo < 0.0 or hi > 2.0 * math.pi:
                raise ValueError(f"{name} 必须在 [0, 2π) 内，得到 ({lo}, {hi})")
        if self.tof_range[0] >= self.tof_range[1]:
            raise ValueError(f"tof_range[0] < tof_range[1] 必须成立，得到 {self.tof_range}")
        if self.tli_speed_factor <= 0.0:
            raise ValueError(f"tli_speed_factor 必须 > 0，得到 {self.tli_speed_factor}")
        if self.perilune_alt_min >= self.perilune_alt_max:
            raise ValueError(
                f"perilune_alt_min < perilune_alt_max 必须成立，"
                f"得到 ({self.perilune_alt_min}, {self.perilune_alt_max})"
            )


@dataclass
class WsbCandidate:
    """单个 WSB 候选解（无动力月球飞越 + BCR4BP 太阳摄动）。

    飞越段 Δv = 0，总 Δv = 出发脉冲 + 到达脉冲。
    ``dv_departure`` / ``dv_arrival`` / ``total_dv`` 均为无量纲
    （× 系统特征速度得 km/s；产自 BCR4BP 搜索）。
    """

    sun_phase0: float
    departure_phase: float
    tof_sec: float
    departure_state: np.ndarray
    perilune_state: np.ndarray
    perilune_alt_km: float
    perilune_time_dim: float
    arrival_state: np.ndarray
    h2_kepler: float
    dv_departure: float
    dv_arrival: float
    total_dv: float
    arrival_time_dim: float
    status: ConvergenceState
    cause: FailureCause
    message: str

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


def compute_kepler_energy_moon(state: np.ndarray, mu: float) -> float:
    """BCR4BP 旋转系中相对月球的开普勒能量（无量纲）。

    速度从旋转系转换到惯性系并减去月球惯性速度，得到相对月球的速度：

        ``v_rel = (vx - y, vy + x - (1-μ), vz)``

    ``H₂ = 0.5 * |v_rel|² - μ / |r - r_moon|``

    符号约定（Belbruno 2010 Eq 2.8）：
        H₂ < 0: 弹道捕获（束缚轨道，无需制动脉冲即被月球束缚）
        H₂ = 0: WSB 边界（抛物线）
        H₂ > 0: 双曲飞越（超逃逸速度）

    Args:
        state: 旋转系无量纲状态 (6,)，[x, y, z, vx, vy, vz]
        mu: 地月质量参数

    Returns:
        无量纲开普勒能量
    """
    x, y, z, vx, vy, vz = state
    moon_x = 1.0 - mu
    rx, ry, rz = x - moon_x, y, z
    r_moon = math.sqrt(rx * rx + ry * ry + rz * rz)

    # 旋转系→惯性系速度修正，减去月球惯性速度 (0, 1-μ, 0)
    rel_vx = vx - y
    rel_vy = vy + x - moon_x
    rel_vz = vz
    v2 = rel_vx * rel_vx + rel_vy * rel_vy + rel_vz * rel_vz

    return 0.5 * v2 - mu / r_moon


def search_wsb_trajectories(
    departure_state: np.ndarray,
    target_state: np.ndarray,
    system: BCR4BPSystem,
    params: WsbSearchParams | None = None,
    *,
    backend: Literal["rust", "python"] = "rust",
    parallel: bool | None = None,
    n_workers: int | None = None,
    progress_callback: Callable[[int], Any] | None = None,
) -> CandidateSearchResult[WsbCandidate]:
    """WSB 弹道三维网格搜索。

    默认 ``backend='rust'``：BCR4BP 传播、近月点检测、候选筛选和 Rayon
    并行均在 Rust 内完成。``backend='python'`` 仅供显式等价性对照，绝不
    在 Rust 扩展缺失或运行失败时自动回退。
    """
    if backend not in ("rust", "python"):
        raise ValueError("backend 须为 'rust' 或 'python'")
    if backend == "rust":
        return _search_wsb_trajectories_rust(
            departure_state,
            target_state,
            system,
            params,
            parallel=parallel,
            n_workers=n_workers,
            progress_callback=progress_callback,
        )
    return _search_wsb_trajectories_python(
        departure_state,
        target_state,
        system,
        params,
        parallel=parallel,
        n_workers=n_workers,
        progress_callback=progress_callback,
    )


def _search_wsb_trajectories_rust(
    departure_state: np.ndarray,
    target_state: np.ndarray,
    system: BCR4BPSystem,
    params: WsbSearchParams | None,
    *,
    parallel: bool | None,
    n_workers: int | None,
    progress_callback: Callable[[int], Any] | None,
) -> CandidateSearchResult[WsbCandidate]:
    """把 WSB POD 输入交给 Rust 核，再恢复现有领域结果对象。"""
    from ...integrators import wsb_search_rust

    if params is None:
        params = WsbSearchParams()
    if system.characteristic_length is None or system.characteristic_time is None:
        raise ValueError("system 必须设置 characteristic_length 与 characteristic_time")

    # 搜索级容差直接由 WsbSearchParams 提供；实际传播由 Rust 核完成。
    candidates, n_propagation_failures, n_perilune_in_window = wsb_search_rust(
        departure_state,
        target_state,
        mu=float(system.mu),
        mu_sun=float(system.sun_mass),
        sun_distance=float(system.sun_distance),
        sun_angular_rate=float(system.sun_angular_rate),
        sun_phase_range=params.sun_phase_range,
        n_sun_phase=params.n_sun_phase,
        departure_phase_range=params.departure_phase_range,
        n_departure_phase=params.n_departure_phase,
        tof_range_sec=(
            params.tof_range[0] * SECONDS_PER_DAY,
            params.tof_range[1] * SECONDS_PER_DAY,
        ),
        n_tof=params.n_tof,
        perilune_alt_range_km=(params.perilune_alt_min, params.perilune_alt_max),
        max_total_dv=params.max_total_dv,
        h2_energy_threshold=params.h2_energy_threshold,
        tli_speed_factor=params.tli_speed_factor,
        n_propagation_samples=params.n_propagation_samples,
        rtol=params.rtol,
        atol=params.atol,
        max_step=math.inf,
        max_steps=params.max_steps,
        secondary_radius_km=R_MOON_KM,
        characteristic_length_km=float(system.characteristic_length),
        characteristic_time_sec=float(system.characteristic_time),
        parallel=parallel,
        n_workers=n_workers,
        progress_callback=progress_callback,
    )
    wrapped = tuple(
        WsbCandidate(
            **candidate,
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message="找到 WSB 候选",
        )
        for candidate in candidates
    )
    if wrapped:
        return CandidateSearchResult(
            wrapped,
            ConvergenceState.CONVERGED,
            FailureCause.NONE,
            "找到 WSB 候选",
        )
    n_tasks = params.n_sun_phase * params.n_tof * params.n_departure_phase
    if n_propagation_failures == n_tasks:
        return CandidateSearchResult(
            (),
            ConvergenceState.DIVERGED,
            FailureCause.DIVERGENCE_DETECTED,
            "全部 WSB 网格点传播失败",
        )
    # 筛选漏斗诊断：零候选时报告卡在哪一环，而非笼统的
    # "未找到可行候选"。近月点高度窗无命中是几何/参数化问题；
    # 有命中但零候选则是 H₂/Δv 筛选拦截。
    funnel = (
        f"搜索未找到可行候选：传播失败 {n_propagation_failures}/{n_tasks}，"
        f"近月点高度窗命中 {n_perilune_in_window}，"
        f"最终候选 0（后续 H₂/Δv 筛选拦截）"
        if n_perilune_in_window > 0
        else (
            f"搜索未找到可行候选：传播失败 {n_propagation_failures}/{n_tasks}，"
            f"无轨迹的近月点进入 [{params.perilune_alt_min:.0f}, "
            f"{params.perilune_alt_max:.0f}] km 高度窗"
        )
    )
    return CandidateSearchResult(
        (),
        ConvergenceState.INFEASIBLE,
        FailureCause.NO_INTERSECTION,
        funnel,
    )


def _search_wsb_trajectories_python(
    departure_state: np.ndarray,
    target_state: np.ndarray,
    system: BCR4BPSystem,
    params: WsbSearchParams | None = None,
    *,
    parallel: bool | None = None,
    n_workers: int | None = None,
    progress_callback: Callable[[int], Any] | None = None,
) -> CandidateSearchResult[WsbCandidate]:
    """WSB 弹道并行网格搜索。

    搜索空间：sun_phase × departure_phase × tof 三维网格。
    ProcessPoolExecutor 并行化，每个 (sun_phase, tof) 独立。

    对每个 (sun_phase, departure_phase, tof) 组合：
    1. 构造 BCR4BP 系统（sun_phase0 = sun_phase）
    2. 从停泊轨道出发，沿 departure_phase 方向施加 TLI 脉冲
    3. BCR4BP 前向传播 tof 时间，检测近月点
    4. 计算 H₂（相对月球开普勒能量），H₂ < threshold 的保留（弹道捕获候选）
    5. Δv_dep + Δv_arr < max_total_dv 的保留为候选

    Args:
        departure_state: 旋转系无量纲出发态 (6,)
        target_state: 旋转系无量纲目标态 (6,)
        system: BCR4BP 系统（提供 mu 和特征尺度）
        params: 搜索参数

    Returns:
        带最终状态的候选搜索结果；可按序列方式读取候选。
    """
    if params is None:
        params = WsbSearchParams()
    mu = system.mu
    du_km = system.characteristic_length
    if du_km is None:
        raise ValueError("system.characteristic_length must be set")

    # 出发态速度参数化：departure_phase 是出发点在停泊轨道上的
    # 滑行角（绕地球旋转 r0 与 v_park，改变月地几何），TLI 脉冲沿切向施加，
    # 速度大小 v_esc * tli_speed_factor。若把 departure_phase 用作 TLI
    # 方向角，切向发射的远地点背向月球，无法以低 Δv 命中低空近月点。
    r0 = departure_state[:3].copy()
    v_park = departure_state[3:].copy()
    r0_norm = np.linalg.norm(r0)
    v_esc = math.sqrt(2.0 * (1.0 - mu) / r0_norm)
    v_tli = v_esc * params.tli_speed_factor
    if np.linalg.norm(v_park) < 1e-12:
        raise ValueError("departure_state 速度分量接近零，无法确定切向 TLI 方向")

    # 目标轨道半径
    r_target = float(np.linalg.norm(target_state[:3]))

    # 搜索网格
    sun_phase_grid = np.linspace(
        params.sun_phase_range[0],
        params.sun_phase_range[1],
        params.n_sun_phase,
        endpoint=False,
    )
    char_time = system.characteristic_time
    if char_time is None:
        raise ValueError("system.characteristic_time must be set")
    tof_grid_sec = np.linspace(
        params.tof_range[0] * SECONDS_PER_DAY,
        params.tof_range[1] * SECONDS_PER_DAY,
        params.n_tof,
    )

    tasks = [(float(sp), float(tof_sec)) for sp in sun_phase_grid for tof_sec in tof_grid_sec]

    worker_args = [
        (
            params,
            departure_state,
            r0,
            v_park,
            v_tli,
            target_state,
            r_target,
            mu,
            du_km,
            char_time,
            float(system.sun_mass),
            float(system.sun_distance),
            float(system.sun_angular_rate),
            sun_phase,
            tof_sec,
        )
        for sun_phase, tof_sec in tasks
    ]
    all_candidates: list[WsbCandidate] = []
    n_propagation_failures = 0
    if parallel is False or n_workers == 1:
        worker_results = [_wsb_worker(*args) for args in worker_args]
    else:
        # spawn 启动子进程：xdist 并行 worker 本身是多线程，fork 出的
        # 子进程继承父进程锁状态，multiprocessing 在 pytest-xdist 下实测会
        # futex 死锁；spawn 重新初始化解释器，无继承锁，安全。
        with ProcessPoolExecutor(
            max_workers=n_workers if n_workers is not None else os.cpu_count(),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(_wsb_worker, *args) for args in worker_args]
            worker_results = [future.result() for future in futures]
    for candidates, propagation_failures in worker_results:
        all_candidates.extend(candidates)
        n_propagation_failures += propagation_failures
        if progress_callback is not None:
            progress_callback(1)

    all_candidates.sort(key=lambda c: c.total_dv)
    if all_candidates:
        return CandidateSearchResult(
            tuple(all_candidates),
            ConvergenceState.CONVERGED,
            FailureCause.NONE,
            "找到 WSB 候选",
        )
    if n_propagation_failures == len(tasks) * params.n_departure_phase:
        return CandidateSearchResult(
            (),
            ConvergenceState.DIVERGED,
            FailureCause.DIVERGENCE_DETECTED,
            "全部 WSB 网格点传播失败",
        )
    return CandidateSearchResult(
        (),
        ConvergenceState.INFEASIBLE,
        FailureCause.NO_INTERSECTION,
        "搜索未找到可行候选",
    )


def _wsb_worker(
    params: WsbSearchParams,
    departure_state: np.ndarray,
    r0: np.ndarray,
    v_park: np.ndarray,
    v_tli: float,
    target_state: np.ndarray,
    r_target: float,
    mu: float,
    du_km: float,
    char_time: float,
    sun_mass: float,
    sun_distance: float,
    sun_angular_rate: float,
    sun_phase0: float,
    tof_sec: float,
) -> tuple[list[WsbCandidate], int]:
    """单个 (sun_phase, tof) 的 WSB 搜索工作函数。

    在 ProcessPoolExecutor 工作进程中运行。对给定的太阳相位角和
    飞行时间，遍历出发相位角网格，返回所有满足条件的候选。

    出发参数化：每个 departure_phase 把出发点（含停泊
    速度）绕地球旋转该角，TLI 脉冲沿旋转后的切向施加。
    """

    def _rot_z(vec: np.ndarray, angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1], vec[2]])

    bcr4bp_system = BCR4BPSystem(
        mu=mu,
        primary="Earth",
        secondary="Moon",
        sun_mass=sun_mass,
        sun_distance=sun_distance,
        sun_angular_rate=sun_angular_rate,
        sun_phase0=sun_phase0,
    )
    dynamics = BCR4BP_Dynamics(bcr4bp_system)
    # 网格筛选级容差：搜索只需初筛，精度由 _refine_wsb_candidate
    # 的 ThreeBodyLambert 打靶保证；用动力学研究级默认容差会让失败组合
    # 烧到 max_steps 才放弃（单组合秒级），全网格成本不可接受。
    dynamics.rtol = params.rtol
    dynamics.atol = params.atol
    dynamics.max_step = math.inf
    periapsis_section = PoincareSection.periapsis("moon", bcr4bp_system)
    moon_pos = np.array([1.0 - mu, 0.0, 0.0])

    angle_grid = np.linspace(
        params.departure_phase_range[0],
        params.departure_phase_range[1],
        params.n_departure_phase,
        endpoint=False,
    )
    n_samples = params.n_propagation_samples
    tof_dim = tof_sec / char_time
    # max_total_dv 语义为 km/s（WsbSearchParams 文档）：候选 Δv 为无量纲，
    # 阈值按特征速度换算到无量纲域再比较（对齐 LGA，lga._search 的做法）。
    max_total_dv_dim = params.max_total_dv / (du_km / char_time)
    candidates: list[WsbCandidate] = []
    n_propagation_failures = 0

    for angle in angle_grid:
        r_dep = _rot_z(r0, angle)
        v_park_rot = _rot_z(v_park, angle)
        v_dep = v_park_rot / np.linalg.norm(v_park_rot) * v_tli
        x0 = np.concatenate([r_dep, v_dep])
        dv_dep = float(np.linalg.norm(v_dep - v_park_rot))

        try:
            t_eval = np.linspace(0.0, tof_dim, n_samples)
            result = dynamics.propagate(x0, (0.0, tof_dim), t_eval=t_eval)
        except PropagationFailure:
            n_propagation_failures += 1
            logger.debug(
                "传播失败：sun_phase=%.3f, angle=%.3f, tof=%.2f",
                sun_phase0,
                angle,
                tof_sec,
            )
            continue

        times = result["time"]
        states = result["states"]

        crossings = detect_crossings(times, states, periapsis_section)
        if not crossings:
            continue

        t_peri, state_peri, idx_peri = crossings[0]
        r_peri_rel = np.linalg.norm(state_peri[:3] - moon_pos)
        alt_km = float(r_peri_rel * du_km - R_MOON_KM)

        if alt_km < params.perilune_alt_min or alt_km > params.perilune_alt_max:
            continue

        h2 = compute_kepler_energy_moon(state_peri, mu)
        if h2 >= params.h2_energy_threshold:
            continue

        # 检测到达目标轨道距离的时刻
        r_traj = np.linalg.norm(states[:, :3], axis=1)
        arrival_state: np.ndarray | None = None
        arrival_time_dim = tof_dim
        for k in range(idx_peri, len(r_traj) - 1):
            r1, r2 = r_traj[k], r_traj[k + 1]
            if (r1 <= r_target <= r2) or (r2 <= r_target <= r1):
                frac = (r_target - r1) / (r2 - r1) if abs(r2 - r1) > 1e-12 else 0.5
                arrival_state = states[k] + frac * (states[k + 1] - states[k])
                arrival_time_dim = times[k] + frac * (times[k + 1] - times[k])
                break

        if arrival_state is None:
            arrival_state = states[-1]

        tof_sec_actual = float(arrival_time_dim * char_time)

        dv_arr = float(np.linalg.norm(arrival_state[3:] - target_state[3:]))
        total_dv = dv_dep + dv_arr

        if total_dv > max_total_dv_dim:
            continue

        candidates.append(
            WsbCandidate(
                sun_phase0=sun_phase0,
                departure_phase=float(angle),
                tof_sec=tof_sec_actual,
                departure_state=x0.copy(),
                perilune_state=state_peri.copy(),
                perilune_alt_km=alt_km,
                perilune_time_dim=float(t_peri),
                arrival_state=arrival_state.copy(),
                h2_kepler=h2,
                dv_departure=dv_dep,
                dv_arrival=dv_arr,
                total_dv=total_dv,
                arrival_time_dim=arrival_time_dim,
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="找到 WSB 候选",
            )
        )

    return candidates, n_propagation_failures


def _refine_wsb_candidate(
    candidate: WsbCandidate,
    system: CR3BP_System,
    dynamics: CR3BP_Dynamics,
    target_state: np.ndarray,
) -> WsbCandidate:
    """用 ThreeBodyLambert 打靶精化 WSB 候选。

    到达段（perilune → target）用 ThreeBodyLambert 修正到达速度。
    打靶未达到收敛状态时返回原始候选。
    """
    from .terminal import StateTerminal
    from .three_body_lambert import ThreeBodyLambert

    char_time = system.characteristic_time
    if char_time is None:
        raise ValueError("system.characteristic_time must be set")

    try:
        shooter = ThreeBodyLambert(dynamics)

        peri_phys = system.dimensionless_to_physical(candidate.perilune_state)
        tof_arrival = (candidate.arrival_time_dim - candidate.perilune_time_dim) * char_time
        if tof_arrival <= 0.0:
            raise ValueError(
                f"到达段剩余时间非正：arrival_time_dim={candidate.arrival_time_dim}, "
                f"perilune_time_dim={candidate.perilune_time_dim}"
            )

        target_phys = system.dimensionless_to_physical(target_state)

        arrival_leg = shooter.solve(
            StateTerminal(peri_phys, 0.0),
            StateTerminal(target_phys, tof_arrival),
            tof_arrival,
            guess="lambert",
        )

        if arrival_leg.status is ConvergenceState.CONVERGED:
            v_arrival_shot = arrival_leg.arcs[-1].states[-1][3:]
            v_target_phys = target_phys[3:]
            # ThreeBodyLambert 解为物理单位 (km/s)：换算回无量纲，
            # 保持 WsbCandidate dv 字段全无量纲语义（对齐 LGA 精化）。
            vu_km_s = system.characteristic_velocity
            if vu_km_s is None or vu_km_s <= 0.0:
                raise ValueError("system.characteristic_velocity must be set")
            dv_arr = float(np.linalg.norm(v_arrival_shot - v_target_phys)) / vu_km_s

            return WsbCandidate(
                sun_phase0=candidate.sun_phase0,
                departure_phase=candidate.departure_phase,
                tof_sec=candidate.tof_sec,
                departure_state=candidate.departure_state,
                perilune_state=candidate.perilune_state,
                perilune_alt_km=candidate.perilune_alt_km,
                perilune_time_dim=candidate.perilune_time_dim,
                arrival_state=candidate.arrival_state,
                h2_kepler=candidate.h2_kepler,
                dv_departure=candidate.dv_departure,
                dv_arrival=dv_arr,
                total_dv=candidate.dv_departure + dv_arr,
                arrival_time_dim=candidate.arrival_time_dim,
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="找到 WSB 候选",
            )
    except (RuntimeError, ValueError, np.linalg.LinAlgError, PropagationFailure):
        # PropagationFailure：打靶内部传播失败（退化候选几何可触发），
        # 与其他打靶失败同义——保留原始候选，不让编排器崩（#566）。
        logger.debug("ThreeBodyLambert 打靶失败，保留原始候选", exc_info=True)

    return WsbCandidate(
        sun_phase0=candidate.sun_phase0,
        departure_phase=candidate.departure_phase,
        tof_sec=candidate.tof_sec,
        departure_state=candidate.departure_state,
        perilune_state=candidate.perilune_state,
        perilune_alt_km=candidate.perilune_alt_km,
        perilune_time_dim=candidate.perilune_time_dim,
        arrival_state=candidate.arrival_state,
        h2_kepler=candidate.h2_kepler,
        dv_departure=candidate.dv_departure,
        dv_arrival=candidate.dv_arrival,
        total_dv=candidate.total_dv,
        arrival_time_dim=candidate.arrival_time_dim,
        status=ConvergenceState.MAX_ITERATIONS,
        cause=FailureCause.MAX_ITERATIONS_REACHED,
        message="WSB 候选精化未收敛",
    )
