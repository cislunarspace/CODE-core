"""LGA 月球引力辅助间接转移：弹道搜索 + 近月飞越处理。

无动力 LGA（unpowered lunar gravity assist）：飞越段 Δv = 0，
总 Δv 仅来自出发脉冲和到达脉冲。

算法来源：Parker & Anderson (2014) §4.3 + Shi et al. (2025) 搜索编排。
CR3BP 直接传播穿越（非 patched conic），近月点检测复用
PoincareSection.periapsis("moon") + detect_crossings()。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ...data.constants import SECONDS_PER_DAY
from ...data.constants.bodies import MOON
from ..dynamics import CR3BP_Dynamics, CR3BP_System
from ..manifold.sections import PoincareSection, detect_crossings

logger = logging.getLogger(__name__)

# 月球半径 (km)
R_MOON_KM: float = MOON.require_mean_radius_km()


@dataclass(frozen=True)
class LgaSearchParams:
    """LGA 弹道搜索参数。

    搜索空间：出发相位角 x 飞行时间（TOF）。
    近月点高度由传播自然决定，不作为独立搜索变量，
    仅用于筛选可行候选（Parker & Anderson 2014 §3.4）。

    Attributes:
        departure_phase_range: 出发相位角范围 (min, max)，弧度，[0, 2pi)
        n_departure_phase: 出发相位角网格点数
        tof_range: 飞行时间范围 (min, max)，天（LGA 典型 15-45 天，Shi et al. 2025）
        n_tof: TOF 网格点数
        perilune_alt_min: 近月点高度下限 (km)，低于此值的候选丢弃（避免撞击月面）
        perilune_alt_max: 近月点高度上限 (km)，高于此值的候选丢弃（飞越不够近）
        max_total_dv: 最大总 Δv 筛选阈值，km/s（超过的候选丢弃）
    """

    departure_phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)
    n_departure_phase: int = 50
    tof_range: tuple[float, float] = (5.0, 45.0)
    n_tof: int = 50
    perilune_alt_min: float = 100.0
    perilune_alt_max: float = 10000.0
    max_total_dv: float = 25.0
    n_propagation_samples: int = 500


@dataclass
class LgaCandidate:
    """单个 LGA 候选解（无动力 LGA）。

    飞越段 Δv = 0（仅利用月球引力），总 Δv = 出发脉冲 + 到达脉冲。
    """

    departure_phase: float
    tof_sec: float
    departure_state: np.ndarray
    perilune_state: np.ndarray
    perilune_alt_km: float
    perilune_time_dim: float
    arrival_state: np.ndarray
    dv_departure: float
    dv_arrival: float
    total_dv: float
    jacobi_departure: float
    jacobi_arrival: float
    arrival_time_dim: float
    converged: bool


def _compute_jacobi(state: np.ndarray, mu: float) -> float:
    """CR3BP Jacobi 常数（Parker 约定）。

    C = 2·U - v²
    U = ½(x² + y²) + (1-μ)/r₁ + μ/r₂
    """
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    u = 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2
    v2 = vx**2 + vy**2 + vz**2
    return float(2.0 * u - v2)


def search_lga_trajectories(
    departure_state: np.ndarray,
    target_state: np.ndarray,
    system: CR3BP_System,
    dynamics: CR3BP_Dynamics,
    params: LgaSearchParams | None = None,
) -> list[LgaCandidate]:
    """LGA 弹道网格搜索。

    搜索空间：出发速度方向角 x 飞行时间（TOF）二维网格。

    出发态为 LEO 停泊轨道（圆轨道速度）。对每个 (angle, tof) 组合：
    1. 从停泊轨道出发，沿 angle 方向施加逃逸速度级的 TLI 脉冲
       （v_tli = v_escape * 1.01，方向由 angle 参数化，确保超逃逸速度）
    2. CR3BP 前向传播 tof 时间，检测近月点（PoincareSection.periapsis("moon")）
    3. 近月点高度在 perilune_alt_range 内的保留
    4. 继续传播，检测轨迹首次到达目标轨道距离（r_target）的时刻
    5. Δv_dep = |v_departure - v_parking|
       Δv_arr = |v_at_target_distance - v_target|
    6. 总 Δv < max_total_dv 的保留为候选

    departure_phase_range 控制出发速度方向角范围（弧度）：
    0 = 纯切向（沿 y 轴），pi/2 = 径向向外（沿 x 轴）。

    Args:
        departure_state: CR3BP 无量纲出发态 (6,)，LEO 停泊轨道
        target_state: CR3BP 无量纲目标态 (6,)
        system: CR3BP 系统
        dynamics: CR3BP 动力学
        params: 搜索参数

    Returns:
        按 total_dv 升序排列的候选列表
    """
    if params is None:
        params = LgaSearchParams()
    mu = system.mu
    du_km = system.characteristic_length
    if du_km is None:
        raise ValueError("system.characteristic_length must be set")
    char_time = system.characteristic_time
    if char_time is None:
        raise ValueError("system.characteristic_time must be set")
    periapsis_section = PoincareSection.periapsis("moon", system)

    # 月球在 CR3BP 旋转系中的位置
    moon_pos = np.array([1.0 - mu, 0.0, 0.0])

    # 目标轨道半径（距质心距离）
    r_target = np.linalg.norm(target_state[:3])

    # 从出发态提取位置和停泊轨道速度
    r0 = departure_state[:3].copy()
    v_park = departure_state[3:].copy()
    r0_norm = np.linalg.norm(r0)

    # 逃逸速度（无量纲）：v_esc = sqrt(2*mu_primary / r)
    v_esc = math.sqrt(2.0 * (1.0 - mu) / r0_norm)
    # TLI 速度：略高于逃逸速度（+1%），确保超逃逸速度
    v_tli = v_esc * 1.01

    # 出发速度方向角网格
    angle_grid = np.linspace(
        params.departure_phase_range[0],
        params.departure_phase_range[1],
        params.n_departure_phase,
        endpoint=False,
    )
    # TOF 网格（无量纲时间）
    tof_grid_dim = np.linspace(
        params.tof_range[0] * SECONDS_PER_DAY / char_time,
        params.tof_range[1] * SECONDS_PER_DAY / char_time,
        params.n_tof,
    )

    # 停泊轨道速度单位向量（切向方向）
    v_park_norm = np.linalg.norm(v_park)
    v_hat = np.array([0.0, 1.0, 0.0]) if v_park_norm < 1e-12 else v_park / v_park_norm

    r_hat = r0 / r0_norm if r0_norm > 1e-12 else np.array([1.0, 0.0, 0.0])

    n_samples = params.n_propagation_samples
    candidates: list[LgaCandidate] = []

    for angle in angle_grid:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        v_dir = cos_a * v_hat + sin_a * r_hat
        v_dep = v_dir * v_tli
        x0 = np.concatenate([r0, v_dep])
        dv_dep = float(np.linalg.norm(v_dep - v_park))

        for tof_dim in tof_grid_dim:
            # 1. 传播 tof 时间，检测近月点
            t_eval = np.linspace(0.0, tof_dim, n_samples)
            try:
                result = dynamics.propagate(x0, (0.0, tof_dim), t_eval=t_eval)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                logger.debug("传播失败：angle=%.3f rad, tof=%.2f", angle, tof_dim)
                continue

            times = result["time"]
            states = result["states"]

            crossings = detect_crossings(times, states, periapsis_section)
            if not crossings:
                continue

            # 取首次近月点
            t_peri, state_peri, idx_peri = crossings[0]
            r_peri_rel = np.linalg.norm(state_peri[:3] - moon_pos)
            alt_km = float(r_peri_rel * du_km - R_MOON_KM)

            if alt_km < params.perilune_alt_min or alt_km > params.perilune_alt_max:
                continue

            # 2. 检测轨迹到达目标轨道距离的时刻
            #    从近月点之后开始搜索
            r_traj = np.linalg.norm(states[:, :3], axis=1)
            arrival_state = None
            arrival_time_dim = tof_dim
            for k in range(idx_peri, len(r_traj) - 1):
                r1, r2 = r_traj[k], r_traj[k + 1]
                if (r1 <= r_target <= r2) or (r2 <= r_target <= r1):
                    frac = (r_target - r1) / (r2 - r1) if abs(r2 - r1) > 1e-12 else 0.5
                    arrival_state = states[k] + frac * (states[k + 1] - states[k])
                    arrival_time_dim = times[k] + frac * (times[k + 1] - times[k])
                    break

            if arrival_state is None:
                # 未到达目标距离：用轨迹末态作为近似
                arrival_state = states[-1]

            tof_sec = float(arrival_time_dim * char_time)

            # Δv 计算
            dv_arr = float(np.linalg.norm(arrival_state[3:] - target_state[3:]))
            total_dv = dv_dep + dv_arr

            if total_dv > params.max_total_dv:
                continue

            # Jacobi 常数
            jac_dep = _compute_jacobi(x0, mu)
            jac_arr = _compute_jacobi(arrival_state, mu)

            candidates.append(
                LgaCandidate(
                    departure_phase=float(angle),
                    tof_sec=tof_sec,
                    departure_state=x0.copy(),
                    perilune_state=state_peri.copy(),
                    perilune_alt_km=alt_km,
                    perilune_time_dim=float(t_peri),
                    arrival_state=arrival_state.copy(),
                    dv_departure=dv_dep,
                    dv_arrival=dv_arr,
                    total_dv=total_dv,
                    jacobi_departure=jac_dep,
                    jacobi_arrival=jac_arr,
                    arrival_time_dim=arrival_time_dim,
                    converged=True,
                )
            )

    candidates.sort(key=lambda c: c.total_dv)
    return candidates


def _refine_lga_candidate(
    candidate: LgaCandidate,
    system: CR3BP_System,
    dynamics: CR3BP_Dynamics,
    target_state: np.ndarray,
) -> LgaCandidate:
    """用 ThreeBodyLambert 打靶精化 LGA 候选。

    分两段打靶：
    1. 到达段：perilune → target（ThreeBodyLambert 打靶修正到达速度）
    2. 打靶后的到达速度更新 Δv 计算。

    如果打靶失败，返回原始候选（converged=False）。
    """
    from .terminal import StateTerminal
    from .three_body_lambert import ThreeBodyLambert

    char_time = system.characteristic_time
    if char_time is None:
        raise ValueError("system.characteristic_time must be set")

    try:
        shooter = ThreeBodyLambert(dynamics)

        peri_phys = system.dimensionless_to_physical(candidate.perilune_state)
        # 到达段的 tof：近月点 → 目标的剩余时间（非出发→到达的总时间）
        tof_arrival = (candidate.arrival_time_dim - candidate.perilune_time_dim) * char_time
        if tof_arrival <= 0.0:
            raise ValueError(
                f"到达段剩余时间非正：arrival_time_dim={candidate.arrival_time_dim}, "
                f"perilune_time_dim={candidate.perilune_time_dim}, tof_arrival={tof_arrival}"
            )

        target_phys = system.dimensionless_to_physical(target_state)

        arrival_leg = shooter.solve(
            StateTerminal(peri_phys, 0.0),
            StateTerminal(target_phys, tof_arrival),
            tof_arrival,
            guess="lambert",
        )

        if arrival_leg.converged:
            # 更新 Δv
            v_arrival_shot = arrival_leg.arcs[-1].states[-1][3:]
            v_target_phys = target_phys[3:]
            dv_arr = float(np.linalg.norm(v_arrival_shot - v_target_phys))

            return LgaCandidate(
                departure_phase=candidate.departure_phase,
                tof_sec=candidate.tof_sec,
                departure_state=candidate.departure_state,
                perilune_state=candidate.perilune_state,
                perilune_alt_km=candidate.perilune_alt_km,
                perilune_time_dim=candidate.perilune_time_dim,
                arrival_state=candidate.arrival_state,
                dv_departure=candidate.dv_departure,
                dv_arrival=dv_arr,
                total_dv=candidate.dv_departure + dv_arr,
                jacobi_departure=candidate.jacobi_departure,
                jacobi_arrival=candidate.jacobi_arrival,
                arrival_time_dim=candidate.arrival_time_dim,
                converged=True,
            )
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        logger.debug("ThreeBodyLambert 打靶失败，保留原始候选", exc_info=True)

    # 打靶失败，返回原始候选
    return LgaCandidate(
        departure_phase=candidate.departure_phase,
        tof_sec=candidate.tof_sec,
        departure_state=candidate.departure_state,
        perilune_state=candidate.perilune_state,
        perilune_alt_km=candidate.perilune_alt_km,
        perilune_time_dim=candidate.perilune_time_dim,
        arrival_state=candidate.arrival_state,
        dv_departure=candidate.dv_departure,
        dv_arrival=candidate.dv_arrival,
        total_dv=candidate.total_dv,
        jacobi_departure=candidate.jacobi_departure,
        jacobi_arrival=candidate.jacobi_arrival,
        arrival_time_dim=candidate.arrival_time_dim,
        converged=False,
    )
