"""转移轨道设计。

按数学类型组织（ADR 0011 迁移，源：``transfer/``）：脉冲路径（lambert/
three_body_lambert/multi_impulse）、自然动力学路径（low_energy/manifold，
覆盖引力辅助数学内核）、低推力路径（low_thrust/）、任务层（search/
optimize/porkchop）。``transfer_orbit.py`` 是编排器：接收 transfer_type
（HMN/LGA/WSB/low_thrust），按枚举选路径组合底层数学模块。

已实现：HMN 霍曼转移、LGA 月球引力辅助、WSB 太阳引力辅助、小推力转移。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...data.constants import SECONDS_PER_DAY
from ...data.templates import ConvergenceState, FailureCause
from ..forces import PointMassGravity
from ..results import CandidateSearchResult, ResultStatus, StageRecord
from .config import (
    TransferArc,
    TransferConfig,
    TransferOptimizationResult,
    TransferSolution,
)
from .hohmann import (
    MU_EARTH,
    R_EARTH,
    TliParams,
    construct_departure_state,
    ephemeris_shoot_transfer,
    hohmann_delta_v,
    hohmann_tof,
    scan_lambert_delta_v,
)
from .lambert import LambertSolution, solve_lambert, solve_lambert_batch
from .lga import LgaCandidate, LgaSearchParams, search_lga_trajectories
from .low_energy import PatchCandidate, design_low_energy_transfer, patch_manifolds
from .lowthrust_collocation import LowThrustCollocation
from .lowthrust_shooting import (
    EngineConfig,
    LowThrustSegment,
    LowThrustShooting,
    LowThrustShootingSolution,
)
from .mission_assessment import MissionAssessment
from .multi_impulse import (
    CoastArc,
    Impulse,
    MultiImpulseTransfer,
    PrimerVectorReport,
)
from .nsga2 import NSGA2Result, nsga2
from .porkchop import ParetoFront, PorkchopData, pareto_front, porkchop
from .propulsion import ImpulsivePropulsion
from .qlaw import qlaw_guess, rv_to_keplerian
from .solution_database import SolutionDatabase
from .terminal import OrbitTerminal, StateTerminal, TerminalCondition
from .three_body_lambert import ThreeBodyLambert
from .thrust_arcs import (
    DEFAULT_THRUST_LEVELS,
    G0_MPS2,
    ThrustArc,
    ThrustArcSequence,
    controls_from_sequence,
    sequence_from_controls,
)
from .transfer import Transfer
from .transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
    optimize_transfer,
    optimize_with_copt,
)
from .transfer_search import (
    DEFAULT_MIN_DISTANCE_THRESHOLD_DU,
    TransferSearch,
    load_orbit_from_json,
)
from .wsb import WsbCandidate, WsbSearchParams, search_wsb_trajectories

__all__ = [
    "TransferSearch",
    "Transfer",
    "TransferConfig",
    "TransferOptimizationResult",
    "DROTRONLPOptimizer",
    "NLPOptimizationVariables",
    "DEFAULT_MIN_DISTANCE_THRESHOLD_DU",
    "load_orbit_from_json",
    "optimize_transfer",
    "optimize_with_copt",
    "ImpulsivePropulsion",
    "TerminalCondition",
    "OrbitTerminal",
    "StateTerminal",
    "LambertSolution",
    "solve_lambert",
    "solve_lambert_batch",
    "PorkchopData",
    "porkchop",
    "ParetoFront",
    "pareto_front",
    "NSGA2Result",
    "nsga2",
    "MissionAssessment",
    "SolutionDatabase",
    "ThreeBodyLambert",
    "TransferArc",
    "TransferSolution",
    "CoastArc",
    "Impulse",
    "MultiImpulseTransfer",
    "PrimerVectorReport",
    "PatchCandidate",
    "patch_manifolds",
    "design_low_energy_transfer",
    "EngineConfig",
    "LowThrustSegment",
    "LowThrustShooting",
    "LowThrustShootingSolution",
    "LowThrustCollocation",
    "ThrustArc",
    "ThrustArcSequence",
    "DEFAULT_THRUST_LEVELS",
    "sequence_from_controls",
    "controls_from_sequence",
    "qlaw_guess",
    "rv_to_keplerian",
    "TliParams",
    "construct_departure_state",
    "hohmann_delta_v",
    "hohmann_tof",
    "scan_lambert_delta_v",
    "ephemeris_shoot_transfer",
    "HmnTransferDetails",
    "LgaTransferDetails",
    "LgaSearchParams",
    "LgaCandidate",
    "CandidateSearchResult",
    "WsbTransferDetails",
    "WsbSearchParams",
    "WsbCandidate",
    "LowThrustTransferDetails",
    "transfer_orbit",
]


_DEFAULT_TOF_GRID_POINTS: int = 50
_G0: float = G0_MPS2  # m/s²，标准重力；以 thrust_arcs.G0_MPS2 为准


@dataclass
class TransferDesignResult:
    """转移轨道设计结果。

    Attributes:
        transfer_type: 转移类型（"HMN"/"LGA"/"WSB"/"low_thrust"）。
        delta_v: 总 Δv（km/s）。
        trajectory: 转移轨迹。
        details: 设计细节（弹道参数汇总）。
        stages: 搜索、精化和打靶等可选阶段的执行记录。
        status: 任务最终状态。
        cause: 导致该状态的原因码。
        message: 人类可读诊断。
    """

    transfer_type: str
    delta_v: float
    trajectory: Any
    details: (
        HmnTransferDetails
        | LgaTransferDetails
        | WsbTransferDetails
        | LowThrustTransferDetails
        | dict[str, Any]
    ) = field(default_factory=dict)
    stages: tuple[StageRecord, ...] = ()
    status: ConvergenceState = ConvergenceState.CONVERGED
    cause: FailureCause = FailureCause.NONE
    message: str = "任务完成"

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@dataclass
class HmnTransferDetails:
    """霍曼转移设计细节。

    Attributes:
        tli_epoch: 出发历元（UTC 字符串或 JD_TDB 浮点数）。
        tof_sec: 飞行时间 (秒)。
        r1_km: 出发轨道半径 (km)。
        r2_km: 目标轨道半径 (km)。
        dv1_km_s: 出发 Δv (km/s)。
        dv2_km_s: 到达 Δv (km/s)。
        departure_state: ECI 出发状态 (6,)。
        delta_v_theory: 理论 Δv (dv1, dv2)。
    """

    tli_epoch: float | str
    tof_sec: float
    r1_km: float
    r2_km: float
    dv1_km_s: float
    dv2_km_s: float
    departure_state: NDArray[np.float64]
    delta_v_theory: tuple[float, float]


@dataclass
class LgaTransferDetails:
    """LGA 月球引力辅助转移设计细节。"""

    tli_epoch: float | str
    tof_sec: float
    perilune_alt_km: float
    perilune_vel_km_s: float
    perilune_state: np.ndarray
    dv_departure_km_s: float
    dv_arrival_km_s: float
    jacobi_departure: float
    jacobi_arrival: float
    n_candidates_searched: int
    n_candidates_feasible: int
    status: ConvergenceState
    cause: FailureCause
    message: str
    search_params: LgaSearchParams

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@dataclass
class WsbTransferDetails:
    """WSB 太阳引力辅助转移设计细节。"""

    tli_epoch: float | str
    tof_sec: float
    perilune_alt_km: float
    perilune_vel_km_s: float
    perilune_state: np.ndarray
    h2_kepler: float
    dv_departure_km_s: float
    dv_arrival_km_s: float
    n_candidates_searched: int
    n_candidates_feasible: int
    status: ConvergenceState
    cause: FailureCause
    message: str
    search_params: WsbSearchParams

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


def _equivalent_delta_v(m0: float, mf: float, isp: float) -> float:
    """Tsiolkovsky 方程：Δv = Isp·g₀·ln(m0/mf)，单位 km/s。

    Args:
        m0: 初始质量 (kg)。
        mf: 末态质量 (kg)。
        isp: 比冲 (s)。

    Returns:
        等效 Δv (km/s)。
    """
    return isp * _G0 * math.log(m0 / mf) / 1000.0


@dataclass
class LowThrustTransferDetails:
    """小推力转移设计细节。

    Attributes:
        engine: 推进配置。
        initial_mass: 初始质量 (kg)。
        final_mass: 末态质量 (kg)。
        fuel_consumed: 燃料消耗 (kg)。
        equivalent_delta_v: 等效 Δv (km/s)，Tsiolkovsky 方程反算。
        n_segments: 求解器段数。
        solver_method: 求解方法 ("shooting" / "collocation")。
        status: 求解最终状态。
        cause: 求解最终原因。
        message: 求解器消息。
        n_iter: 迭代次数。
        terminal_residual_r: 终端位置残差 (km)。
        terminal_residual_v: 终端速度残差 (km/s)。
        time: 采样时间序列 (M,)，SPICE et 秒。
        states_7d: 7D 状态序列 (M, 7) [x,y,z,vx,vy,vz,m]。
        segments: 各段常量控制。
        qlaw_q_history: Q-law Q 值历史（仅 solve_from_qlaw 时非空）。
    """

    engine: EngineConfig
    initial_mass: float
    final_mass: float
    fuel_consumed: float
    equivalent_delta_v: float
    n_segments: int
    solver_method: str  # "shooting" | "collocation"
    status: ConvergenceState
    cause: FailureCause
    message: str
    n_iter: int
    terminal_residual_r: float  # km
    terminal_residual_v: float  # km/s
    time: NDArray[np.float64]
    states_7d: NDArray[np.float64]
    segments: tuple[LowThrustSegment, ...]
    qlaw_q_history: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


def transfer_orbit(
    transfer_type: str,
    *,
    target_ephemeris: Any = None,
    tli_params: TliParams | None = None,
    tof_range: tuple[float, float] | None = None,
    target_orbit_radius_km: float | None = None,
    dynamics: Any = None,
    lga_search_params: LgaSearchParams | None = None,
    wsb_search_params: WsbSearchParams | None = None,
    engine_config: EngineConfig | None = None,
    initial_mass: float | None = None,
    n_segments: int = 10,
    target_oe: tuple[float, float, float] | None = None,
    solver_method: str = "shooting",
    duration_days: float = 30.0,
    departure_state: NDArray[np.float64] | None = None,
    target_state: NDArray[np.float64] | None = None,
    system: Any = None,
    forces: Any = None,
    **kwargs,
) -> TransferDesignResult:
    """端到端转移轨道设计（编排器）。

    Args:
        transfer_type: "HMN"（直接）/ "LGA"（月球引力辅助）/ "WSB"（太阳引力辅助）/
            "low_thrust"（小推力）。
        target_ephemeris: 目标轨道星历（FR1 产物）。
        tli_params: 地球停泊轨道参数（TLI 高度/倾角/航迹角）。
        tof_range: 飞行时间范围（天）。
        target_orbit_radius_km: 目标轨道半径 (km)，HMN 转移必需。
        dynamics: 动力学对象（可选），用于 ephemeris 打靶修正。
        lga_search_params: LGA 搜索参数（可选）。
        wsb_search_params: WSB 搜索参数（可选）。
        engine_config: 推进配置（小推力转移必需）。
        initial_mass: 初始质量 kg（小推力转移必需）。
        n_segments: 求解器段数（小推力，默认 10）。
        target_oe: Q-law 目标 ``(a_T, e_T, i_T)`` （小推力可选）。
        solver_method: 求解方法 ``"shooting"`` / ``"collocation"`` （小推力，默认 ``"shooting"``）。
        duration_days: 飞行时间（天）（小推力，默认 30.0）。
        departure_state: 小推力出发状态 ``[r, v]`` (6,)，km / km/s（小推力可选）。
        target_state: 小推力目标末态 ``[r, v]`` (6,)，km / km/s（小推力可选）。
        system: 动力学系统（小推力可选，默认纯二体）。
        forces: 非推力力模型列表（小推力可选）。

    Returns:
        TransferDesignResult: 转移轨道设计结果。

    Raises:
        NotImplementedError: 编排器实现未完成（未知的 transfer_type）。
        ValueError: 转移类型缺少必要参数。
    """
    if transfer_type == "HMN":
        return _transfer_orbit_hmn(
            tli_params,
            target_orbit_radius_km,
            tof_range,
            dynamics=dynamics,
            target_ephemeris=target_ephemeris,
        )
    if transfer_type == "LGA":
        return _transfer_orbit_lga(
            tli_params=tli_params,
            target_ephemeris=target_ephemeris,
            search_params=lga_search_params,
            dynamics=dynamics,
        )
    if transfer_type == "WSB":
        return _transfer_orbit_wsb(
            tli_params=tli_params,
            target_ephemeris=target_ephemeris,
            search_params=wsb_search_params,
        )
    if transfer_type == "low_thrust":
        if engine_config is None:
            raise ValueError("low_thrust 转移需要 engine_config")
        if initial_mass is None:
            raise ValueError("low_thrust 转移需要 initial_mass")
        return _transfer_orbit_low_thrust(
            tli_params=tli_params,
            target_ephemeris=target_ephemeris,
            engine_config=engine_config,
            initial_mass=initial_mass,
            n_segments=n_segments,
            target_oe=target_oe,
            solver_method=solver_method,
            duration_days=duration_days,
            departure_state=departure_state,
            target_state=target_state,
            system=system,
            forces=forces,
        )
    raise NotImplementedError(f"transfer_orbit('{transfer_type}') 实现未完成（能力在规划中）")


def _extract_target_state(target_ephemeris: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """从 target_ephemeris 提取目标位置和速度。

    支持三种输入格式：

    - numpy ndarray (n, 6)：取最后一行。
    - NominalOrbit（有 .states 属性，形状 (n, 6)）：取最后一行。
    - EphemerisTable（有 position_km (n, 3) 和 velocity_mps (n, 3)）：
      取最后一行，速度从 m/s 转换为 km/s。

    Returns:
        (r_target, v_target)，单位 km 和 km/s。
    """
    if isinstance(target_ephemeris, np.ndarray):
        last_state = target_ephemeris[-1]
        return last_state[:3].copy(), last_state[3:6].copy()
    # NominalOrbit: .states 形状 (n, 6)
    if hasattr(target_ephemeris, "states"):
        last_state = np.asarray(target_ephemeris.states[-1])
        return last_state[:3].copy(), last_state[3:6].copy()
    # EphemerisTable: .position_km (n, 3), .velocity_mps (n, 3)
    if hasattr(target_ephemeris, "position_km") and hasattr(target_ephemeris, "velocity_mps"):
        r_target = np.asarray(target_ephemeris.position_km[-1], dtype=np.float64)
        v_target = np.asarray(target_ephemeris.velocity_mps[-1], dtype=np.float64) / 1000.0
        return r_target.copy(), v_target.copy()
    raise TypeError(
        f"不支持的 target_ephemeris 类型：{type(target_ephemeris).__name__}，"
        "期望 ndarray (n,6)、NominalOrbit 或 EphemerisTable"
    )


def _transfer_orbit_lga(
    tli_params: TliParams | None,
    target_ephemeris: Any,
    search_params: LgaSearchParams | None,
    dynamics: Any = None,
) -> TransferDesignResult:
    """LGA 月球引力辅助转移编排。

    流程：
    1. TliParams → ECI 出发态
    2. 目标星历 → 目标态
    3. ECI → CR3BP 无量纲
    4. search_lga_trajectories() 网格搜索
    5. 取最优候选
    6. _refine_lga_candidate() ThreeBodyLambert 打靶精化
    7. 物理单位换算 + 结果汇总
    """
    if tli_params is None:
        raise ValueError("LGA 转移需要 tli_params")
    if target_ephemeris is None:
        raise ValueError("LGA 转移需要 target_ephemeris")

    from ..dynamics import CR3BP_Dynamics, CR3BP_System

    # 地月 CR3BP 系统
    MU_EM = 1.21506683e-2
    system = CR3BP_System(mu=MU_EM, primary="Earth", secondary="Moon")._with_default_scales()
    cr3bp_dynamics = CR3BP_Dynamics(system)

    # 1. ECI 出发态
    r0, v0 = construct_departure_state(tli_params)
    departure_phys = np.concatenate([r0, v0])
    departure_dim = system.physical_to_dimensionless(departure_phys)

    # 2. 目标态
    r_target, v_target = _extract_target_state(target_ephemeris)
    target_phys = np.concatenate([r_target, v_target])
    target_dim = system.physical_to_dimensionless(target_phys)

    # 3. LGA 搜索
    candidates = search_lga_trajectories(
        departure_dim, target_dim, system, cr3bp_dynamics, search_params
    )

    params = search_params if search_params is not None else LgaSearchParams()

    n_searched = params.n_departure_phase * params.n_tof
    n_feasible = len(candidates)

    if not candidates:
        warnings.warn("LGA 搜索未找到可行候选，返回零结果", stacklevel=2)
        details = LgaTransferDetails(
            tli_epoch=tli_params.epoch,
            tof_sec=0.0,
            perilune_alt_km=0.0,
            perilune_vel_km_s=0.0,
            perilune_state=np.zeros(6),
            dv_departure_km_s=0.0,
            dv_arrival_km_s=float("inf"),
            jacobi_departure=0.0,
            jacobi_arrival=0.0,
            n_candidates_searched=n_searched,
            n_candidates_feasible=0,
            status=candidates.status,
            cause=candidates.cause,
            message=candidates.message,
            search_params=params,
        )
        return TransferDesignResult(
            transfer_type="LGA",
            delta_v=float("inf"),
            trajectory=None,
            details=details,
            status=details.status,
            cause=details.cause,
            message=details.message,
            stages=(
                StageRecord(
                    "search",
                    applicable=True,
                    executed=True,
                    result_status=candidates.status,
                    message=candidates.message,
                ),
                StageRecord("refinement", applicable=True, executed=False, result_status=None),
                StageRecord("shooting", applicable=True, executed=False, result_status=None),
            ),
        )

    # 4. 取最优候选
    best = candidates[0]

    # 5. ThreeBodyLambert 打靶精化
    from .lga import _refine_lga_candidate

    refined = _refine_lga_candidate(best, system, cr3bp_dynamics, target_dim)

    # 6. 物理单位换算
    perilune_phys = system.dimensionless_to_physical(refined.perilune_state)
    perilune_vel = float(np.linalg.norm(perilune_phys[3:]))

    details = LgaTransferDetails(
        tli_epoch=tli_params.epoch,
        tof_sec=refined.tof_sec,
        perilune_alt_km=refined.perilune_alt_km,
        perilune_vel_km_s=perilune_vel,
        perilune_state=perilune_phys,
        dv_departure_km_s=refined.dv_departure,
        dv_arrival_km_s=refined.dv_arrival,
        jacobi_departure=refined.jacobi_departure,
        jacobi_arrival=refined.jacobi_arrival,
        n_candidates_searched=n_searched,
        n_candidates_feasible=n_feasible,
        status=refined.status,
        cause=refined.cause,
        message=refined.message,
        search_params=params,
    )

    return TransferDesignResult(
        transfer_type="LGA",
        delta_v=refined.total_dv,
        trajectory=None,
        details=details,
        status=refined.status,
        cause=refined.cause,
        message=refined.message,
        stages=(
            StageRecord(
                "search",
                applicable=True,
                executed=True,
                result_status=ConvergenceState.CONVERGED,
                message="找到可行候选",
            ),
            StageRecord("refinement", applicable=True, executed=True, result_status=refined.status),
            StageRecord(
                "shooting",
                applicable=True,
                executed=True,
                result_status=refined.status,
                message=refined.message,
            ),
        ),
    )


def _transfer_orbit_wsb(
    tli_params: TliParams | None,
    target_ephemeris: Any,
    search_params: WsbSearchParams | None,
) -> TransferDesignResult:
    """WSB 太阳引力辅助转移编排。

    流程：
    1. TliParams → ECI 出发态
    2. 目标星历 → 目标态
    3. ECI → BCR4BP 无量纲（特征尺度与 CR3BP 共用）
    4. search_wsb_trajectories() 并行网格搜索
    5. 取最优候选
    6. _refine_wsb_candidate() ThreeBodyLambert 打靶精化
    7. 物理单位换算 + 结果汇总
    """
    if tli_params is None:
        raise ValueError("WSB 转移需要 tli_params")
    if target_ephemeris is None:
        raise ValueError("WSB 转移需要 target_ephemeris")

    from ..dynamics import CR3BP_Dynamics, CR3BP_System
    from ..dynamics.bcr4bp_system import BCR4BPSystem
    from .wsb import _refine_wsb_candidate

    # BCR4BP 系统（搜索用，sun_phase0 在 worker 中逐个构造）
    MU_EM = 1.21506683e-2
    bcr4bp_system = BCR4BPSystem.earth_moon()

    # 1. ECI 出发态
    r0, v0 = construct_departure_state(tli_params)
    departure_phys = np.concatenate([r0, v0])
    departure_dim = bcr4bp_system.physical_to_dimensionless(departure_phys)

    # 2. 目标态
    r_target, v_target = _extract_target_state(target_ephemeris)
    target_phys = np.concatenate([r_target, v_target])
    target_dim = bcr4bp_system.physical_to_dimensionless(target_phys)

    # 3. WSB 搜索（并行）
    candidates = search_wsb_trajectories(departure_dim, target_dim, bcr4bp_system, search_params)

    params = search_params if search_params is not None else WsbSearchParams()
    n_searched = params.n_sun_phase * params.n_departure_phase * params.n_tof
    n_feasible = len(candidates)

    if not candidates:
        warnings.warn("WSB 搜索未找到可行候选，返回零结果", stacklevel=2)
        details = WsbTransferDetails(
            tli_epoch=tli_params.epoch,
            tof_sec=0.0,
            perilune_alt_km=0.0,
            perilune_vel_km_s=0.0,
            perilune_state=np.zeros(6),
            h2_kepler=0.0,
            dv_departure_km_s=0.0,
            dv_arrival_km_s=float("inf"),
            n_candidates_searched=n_searched,
            n_candidates_feasible=0,
            status=candidates.status,
            cause=candidates.cause,
            message=candidates.message,
            search_params=params,
        )
        return TransferDesignResult(
            transfer_type="WSB",
            delta_v=float("inf"),
            trajectory=None,
            details=details,
            status=details.status,
            cause=details.cause,
            message=details.message,
            stages=(
                StageRecord(
                    "search",
                    applicable=True,
                    executed=True,
                    result_status=candidates.status,
                    message=candidates.message,
                ),
                StageRecord("refinement", applicable=True, executed=False, result_status=None),
                StageRecord("shooting", applicable=True, executed=False, result_status=None),
            ),
        )

    # 4. 取最优候选
    best = candidates[0]

    # 5. ThreeBodyLambert 打靶精化（CR3BP 到达段）
    cr3bp_system = CR3BP_System(mu=MU_EM, primary="Earth", secondary="Moon")._with_default_scales()
    cr3bp_dynamics = CR3BP_Dynamics(cr3bp_system)
    refined = _refine_wsb_candidate(best, cr3bp_system, cr3bp_dynamics, target_dim)

    # 6. 物理单位换算
    perilune_phys = bcr4bp_system.dimensionless_to_physical(refined.perilune_state)
    perilune_vel = float(np.linalg.norm(perilune_phys[3:]))

    details = WsbTransferDetails(
        tli_epoch=tli_params.epoch,
        tof_sec=refined.tof_sec,
        perilune_alt_km=refined.perilune_alt_km,
        perilune_vel_km_s=perilune_vel,
        perilune_state=perilune_phys,
        h2_kepler=refined.h2_kepler,
        dv_departure_km_s=refined.dv_departure,
        dv_arrival_km_s=refined.dv_arrival,
        n_candidates_searched=n_searched,
        n_candidates_feasible=n_feasible,
        status=refined.status,
        cause=refined.cause,
        message=refined.message,
        search_params=params,
    )

    return TransferDesignResult(
        transfer_type="WSB",
        delta_v=refined.total_dv,
        trajectory=None,
        details=details,
        status=refined.status,
        cause=refined.cause,
        message=refined.message,
        stages=(
            StageRecord(
                "search",
                applicable=True,
                executed=True,
                result_status=ConvergenceState.CONVERGED,
                message="找到可行候选",
            ),
            StageRecord("refinement", applicable=True, executed=True, result_status=refined.status),
            StageRecord(
                "shooting",
                applicable=True,
                executed=True,
                result_status=refined.status,
                message=refined.message,
            ),
        ),
    )


def _transfer_orbit_low_thrust(
    tli_params: TliParams | None,
    target_ephemeris: Any,
    engine_config: EngineConfig,
    initial_mass: float,
    n_segments: int = 10,
    *,
    target_oe: tuple[float, float, float] | None = None,
    solver_method: str = "shooting",
    duration_days: float = 30.0,
    departure_state: np.ndarray | None = None,
    target_state: np.ndarray | None = None,
    system: Any = None,
    forces: Any = None,
) -> TransferDesignResult:
    """小推力转移编排。

    流程：
    1. 出发状态：优先 ``departure_state``，否则 ``construct_departure_state(tli_params)``。
    2. 目标状态：优先 ``target_state``，否则 ``_extract_target_state(target_ephemeris)``。
    3. 动力学系统/力模型：优先传入参数，否则构造纯二体。
    4. 构造 ``LowThrustShooting`` 或 ``LowThrustCollocation`` 求解器。
    5. ``solve_from_qlaw()`` Q-law 初猜 + 求解。
    6. 计算终端残差、等效 Δv，返回 ``TransferDesignResult``。

    Args:
        tli_params: 地球停泊轨道参数（TLI 高度/倾角/航迹角）。当 ``departure_state``
            未提供时用于构造出发状态。
        target_ephemeris: 目标轨道星历。当 ``target_state`` 未提供时用于提取目标状态。
        engine_config: 推进配置（最大推力、比冲）。
        initial_mass: 初始质量 (kg)。
        n_segments: 求解器段数。
        target_oe: Q-law 目标 ``(a_T, e_T, i_T)``。默认从目标状态反推圆轨道。
        solver_method: 求解方法 ``"shooting"`` 或 ``"collocation"``。
        duration_days: 飞行时间 (天)。
        departure_state: 出发状态 ``[r, v]`` (6,)，km / km/s。优先于 tli_params。
        target_state: 目标末态 ``[r, v]`` (6,)，km / km/s。优先于 target_ephemeris。
        system: 动力学系统。默认纯二体 ``SimpleNamespace(origin="EARTH")``。
        forces: 非推力力模型列表。默认 ``[PointMassGravity("EARTH", mu=MU_EARTH)]``。

    Returns:
        TransferDesignResult: 转移轨道设计结果，携带 ``LowThrustTransferDetails``。
    """
    # 1. 出发状态
    if departure_state is not None:
        r0 = departure_state[:3]
        v0 = departure_state[3:6]
    else:
        if tli_params is None:
            raise ValueError("low_thrust 转移需要 tli_params 或 departure_state 之一")
        r0, v0 = construct_departure_state(tli_params)

    # 2. 目标状态
    if target_state is not None:
        r_target = target_state[:3]
        v_target = target_state[3:6]
    else:
        if target_ephemeris is None:
            raise ValueError("low_thrust 转移需要 target_ephemeris 或 target_state 之一")
        r_target, v_target = _extract_target_state(target_ephemeris)

    # 3. 动力学系统和力模型
    if system is None:
        system = SimpleNamespace(origin="EARTH")
    if forces is None:
        forces = [PointMassGravity("EARTH", mu=MU_EARTH)]

    # 4. 时间基准
    has_spice = hasattr(system, "spice") and system.spice is not None
    t0 = system.spice.utc_to_et(tli_params.epoch) if has_spice and tli_params is not None else 0.0
    tf = t0 + duration_days * SECONDS_PER_DAY

    # 5. 目标轨道根数（默认圆轨道，从目标状态反推半长轴）
    if target_oe is None:
        r_target_norm = float(np.linalg.norm(r_target))
        v_target_norm = float(np.linalg.norm(v_target))
        energy = v_target_norm**2 / 2.0 - MU_EARTH / r_target_norm
        a_target = -MU_EARTH / (2.0 * energy)
        target_oe = (a_target, 0.0, 0.0)

    # 6. 构造求解器
    initial_state_6 = np.concatenate([r0, v0])
    target_state_6 = np.concatenate([r_target, v_target])

    solver: LowThrustShooting | LowThrustCollocation
    if solver_method == "shooting":
        solver = LowThrustShooting(
            system=system,
            forces=forces,
            engine=engine_config,
            initial_state=initial_state_6,
            initial_mass=initial_mass,
            target_state=target_state_6,
            t0=t0,
            tf=tf,
        )
    elif solver_method == "collocation":
        solver = LowThrustCollocation(
            system=system,
            forces=forces,
            engine=engine_config,
            initial_state=initial_state_6,
            initial_mass=initial_mass,
            target_state=target_state_6,
            t0=t0,
            tf=tf,
        )
    else:
        raise ValueError(
            f"不支持的 solver_method: {solver_method!r}，期望 'shooting' 或 'collocation'"
        )

    # 7. Q-law 初猜 + 求解
    sol: LowThrustShootingSolution = solver.solve_from_qlaw(n_segments, target_oe, forces)

    # 8. 终端残差
    r_final = sol.states[-1, :3]
    v_final = sol.states[-1, 3:6]
    terminal_residual_r = float(np.linalg.norm(r_final - r_target))
    terminal_residual_v = float(np.linalg.norm(v_final - v_target))

    # 9. 等效 Δv
    final_mass = sol.final_mass
    equiv_dv = _equivalent_delta_v(initial_mass, final_mass, engine_config.isp)

    # 10. Q-law Q 值历史（solve_from_qlaw 不返回 q_history，设为 None）
    qlaw_q_history = None

    # 11. 汇总
    details = LowThrustTransferDetails(
        engine=engine_config,
        initial_mass=initial_mass,
        final_mass=final_mass,
        fuel_consumed=sol.fuel_consumed,
        equivalent_delta_v=equiv_dv,
        n_segments=n_segments,
        solver_method=solver_method,
        status=sol.status,
        cause=sol.cause,
        message=sol.message,
        n_iter=sol.n_iter,
        terminal_residual_r=terminal_residual_r,
        terminal_residual_v=terminal_residual_v,
        time=sol.time.astype(np.float64),
        states_7d=sol.states.astype(np.float64),
        segments=sol.segments,
        qlaw_q_history=qlaw_q_history,
    )

    return TransferDesignResult(
        transfer_type="low_thrust",
        delta_v=equiv_dv,
        trajectory=sol.states,
        details=details,
        status=sol.status,
        cause=sol.cause,
        message=sol.message,
        stages=(
            StageRecord("search", applicable=False, executed=False, result_status=None),
            StageRecord("refinement", applicable=False, executed=False, result_status=None),
            StageRecord(
                "shooting",
                applicable=True,
                executed=True,
                result_status=sol.status,
                message=sol.message,
            ),
        ),
    )


def _transfer_orbit_hmn(
    tli_params: TliParams | None,
    target_orbit_radius_km: float | None,
    tof_range: tuple[float, float] | None = None,
    dynamics: Any = None,
    target_ephemeris: Any = None,
) -> TransferDesignResult:
    """HMN 霍曼转移编排：解析解 + 出发状态构造。

    当 ``tof_range`` 提供时，用 Lambert 批量扫描最优 tof；
    否则用霍曼公式计算固定 tof。

    当 ``dynamics`` 提供时，调用 ``ephemeris_shoot_transfer`` 在给定动力学
    模型下修正 Lambert 初猜（多重打靶收敛）。
    """
    if tli_params is None:
        raise ValueError("HMN 转移需要 tli_params")
    if target_orbit_radius_km is None:
        raise ValueError("HMN 转移需要 target_orbit_radius_km")

    r1 = R_EARTH + tli_params.parking_alt_km
    r2 = target_orbit_radius_km

    dv1, dv2 = hohmann_delta_v(r1, r2)
    tof = hohmann_tof(r1, r2)

    r0, v0 = construct_departure_state(tli_params)

    if tof_range is not None:
        tof_min_sec = tof_range[0] * SECONDS_PER_DAY
        tof_max_sec = tof_range[1] * SECONDS_PER_DAY
        tof_grid = np.linspace(tof_min_sec, tof_max_sec, _DEFAULT_TOF_GRID_POINTS)

        # 优先从 target_ephemeris 提取目标状态（RED-2）
        if target_ephemeris is not None:
            r_target, v_target = _extract_target_state(target_ephemeris)
        else:
            # 目标位置：负 x 轴（180° 转移角，与霍曼转移几何一致）
            r_target = np.array([-r2, 0.0, 0.0])
            # 目标速度：圆轨道近似，沿 y 轴切向
            v_target = np.array([0.0, -np.sqrt(MU_EARTH / r2), 0.0])

        optimal_tof, v0_lambert, vf_lambert = scan_lambert_delta_v(
            r0, v0, r_target, v_target, tof_grid
        )
        tof = optimal_tof
        dv1 = float(np.linalg.norm(v0_lambert - v0))
        dv2 = float(np.linalg.norm(vf_lambert - v_target))

    # 当 dynamics 提供时，用 ephemeris 打靶修正 Lambert 初猜
    trajectory: Any = None
    if dynamics is not None:
        t0 = 0.0  # 动力学模型的时间基准（秒）
        shoot_result = ephemeris_shoot_transfer(
            dynamics=dynamics,
            t0=t0,
            r0=r0,
            v0=v0 + np.array([0.0, dv1, 0.0]) if tof_range is None else v0_lambert,
            tof=tof,
        )
        if shoot_result.status is ConvergenceState.CONVERGED:
            # 用打靶收敛的出发状态更新 delta_v 和 departure_state
            v0_shot = shoot_result.state_patch[0, 3:6]
            dv1 = float(np.linalg.norm(v0_shot - v0))
            departure_state = shoot_result.state_patch[0].copy()
            trajectory = shoot_result.state_patch
        else:
            warnings.warn(
                "ephemeris_shoot_transfer 未收敛，回退到 Lambert 解",
                stacklevel=2,
            )
            departure_state = np.concatenate([r0, v0])
            trajectory = None
    else:
        departure_state = np.concatenate([r0, v0])

    details = HmnTransferDetails(
        tli_epoch=tli_params.epoch,
        tof_sec=tof,
        r1_km=r1,
        r2_km=r2,
        dv1_km_s=dv1,
        dv2_km_s=dv2,
        departure_state=departure_state,
        delta_v_theory=(dv1, dv2),
    )

    if dynamics is None:
        status = ConvergenceState.CONVERGED
        cause = FailureCause.NONE
        message = "霍曼转移完成"
    else:
        status = shoot_result.status
        cause = shoot_result.cause
        message = shoot_result.message

    return TransferDesignResult(
        transfer_type="HMN",
        delta_v=dv1 + dv2,
        trajectory=trajectory,
        details=details,
        status=status,
        cause=cause,
        message=message,
        stages=(
            StageRecord(
                "search",
                applicable=tof_range is not None,
                executed=tof_range is not None,
                result_status=ConvergenceState.CONVERGED if tof_range is not None else None,
            ),
            StageRecord("refinement", applicable=False, executed=False, result_status=None),
            StageRecord(
                "shooting",
                applicable=dynamics is not None,
                executed=dynamics is not None,
                result_status=shoot_result.status if dynamics is not None else None,
                message=shoot_result.message if dynamics is not None else "",
            ),
        ),
    )
