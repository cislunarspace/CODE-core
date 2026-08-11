"""WSB 太阳引力辅助间接转移：BCR4BP 弹道搜索 + 到达段精化。

弱稳定边界（Weak Stability Boundary）转移利用地月 BCR4BP 动力学中的
太阳引力摄动，在近月点附近使航天器的相对月球 Kepler 能量 H₂ < 0
（弹道捕获判据——无需制动脉冲即被月球束缚），自然被月球捕获后可由
小量圆化脉冲稳定。总 Δv 仅来自出发脉冲和到达脉冲。

搜索空间：sun_phase × departure_phase × tof 三维网格。
并行化：ProcessPoolExecutor，每个 (sun_phase, tof) 独立。

BCR4BP 旋转系→惯性系速度修正（任务 #259 方案）：
    v_rel_moon = (vx - y, vy + x - (1-μ), vz)
其中 (1-μ, 0, 0) 为月球在旋转系中的位置，减去月球惯性速度
ω × r_moon = (0, 1-μ, 0) 得到相对月球的惯性系速度。
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from ...data.constants import SECONDS_PER_DAY
from ...data.constants.bodies import MOON
from ...data.templates import ConvergenceState, FailureCause
from ..dynamics import BCR4BP_Dynamics, BCR4BPSystem, CR3BP_Dynamics, CR3BP_System
from ..manifold.sections import PoincareSection, detect_crossings
from ..results import ResultStatus

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
        departure_phase_range: 出发相位角范围 (min, max)，弧度，[0, 2pi)
        n_departure_phase: 出发相位角网格点数
        tof_range: 飞行时间范围 (min, max)，天（WSB 典型 90-150 天）
        n_tof: TOF 网格点数
        perilune_alt_min: 近月点高度下限 (km)
        perilune_alt_max: 近月点高度上限 (km)
        max_total_dv: 最大总 Δv 筛选阈值 (km/s)
        h2_energy_threshold: H₂ 能量阈值（无量纲），H₂ < 此值的候选保留（弹道捕获）
        n_propagation_samples: 传播采样点数
    """

    sun_phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)
    n_sun_phase: int = 50
    departure_phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)
    n_departure_phase: int = 50
    tof_range: tuple[float, float] = (90.0, 150.0)
    n_tof: int = 50
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
        if self.perilune_alt_min >= self.perilune_alt_max:
            raise ValueError(
                f"perilune_alt_min < perilune_alt_max 必须成立，"
                f"得到 ({self.perilune_alt_min}, {self.perilune_alt_max})"
            )


@dataclass
class WsbCandidate:
    """单个 WSB 候选解（无动力月球飞越 + BCR4BP 太阳摄动）。

    飞越段 Δv = 0，总 Δv = 出发脉冲 + 到达脉冲。
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
        v_rel = (vx - y, vy + x - (1-μ), vz)

    H₂ = 0.5 * |v_rel|² - μ / |r - r_moon|

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
) -> list[WsbCandidate]:
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
        按 total_dv 升序排列的候选列表
    """
    if params is None:
        params = WsbSearchParams()
    mu = system.mu
    du_km = system.characteristic_length
    if du_km is None:
        raise ValueError("system.characteristic_length must be set")

    # 出发态速度参数化
    r0 = departure_state[:3].copy()
    v_park = departure_state[3:].copy()
    r0_norm = np.linalg.norm(r0)
    v_esc = math.sqrt(2.0 * (1.0 - mu) / r0_norm)
    v_tli = v_esc * 1.01  # 略高于逃逸速度

    v_park_norm = np.linalg.norm(v_park)
    v_hat = np.array([0.0, 1.0, 0.0]) if v_park_norm < 1e-12 else v_park / v_park_norm
    r_hat = r0 / r0_norm if r0_norm > 1e-12 else np.array([1.0, 0.0, 0.0])

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

    all_candidates: list[WsbCandidate] = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(
                _wsb_worker,
                params,
                departure_state,
                r0,
                v_hat,
                r_hat,
                v_tli,
                v_park,
                target_state,
                r_target,
                mu,
                du_km,
                char_time,
                sun_phase,
                tof_sec,
            )
            for sun_phase, tof_sec in tasks
        ]
        for future in futures:
            all_candidates.extend(future.result())

    all_candidates.sort(key=lambda c: c.total_dv)
    return all_candidates


def _wsb_worker(
    params: WsbSearchParams,
    departure_state: np.ndarray,
    r0: np.ndarray,
    v_hat: np.ndarray,
    r_hat: np.ndarray,
    v_tli: float,
    v_park: np.ndarray,
    target_state: np.ndarray,
    r_target: float,
    mu: float,
    du_km: float,
    char_time: float,
    sun_phase0: float,
    tof_sec: float,
) -> list[WsbCandidate]:
    """单个 (sun_phase, tof) 的 WSB 搜索工作函数。

    在 ProcessPoolExecutor 工作进程中运行。对给定的太阳相位角和
    飞行时间，遍历出发相位角网格，返回所有满足条件的候选。
    """
    bcr4bp_system = BCR4BPSystem.earth_moon(sun_phase0=sun_phase0)
    dynamics = BCR4BP_Dynamics(bcr4bp_system)
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
    candidates: list[WsbCandidate] = []

    for angle in angle_grid:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        v_dir = cos_a * v_hat + sin_a * r_hat
        v_dep = v_dir * v_tli
        x0 = np.concatenate([r0, v_dep])
        dv_dep = float(np.linalg.norm(v_dep - v_park))

        try:
            t_eval = np.linspace(0.0, tof_dim, n_samples)
            result = dynamics.propagate(x0, (0.0, tof_dim), t_eval=t_eval)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
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

        if total_dv > params.max_total_dv:
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

    return candidates


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
            dv_arr = float(np.linalg.norm(v_arrival_shot - v_target_phys))

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
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
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
