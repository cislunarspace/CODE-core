"""转移轨道设计。

按数学类型组织（ADR 0011 迁移，源：``transfer/``）：脉冲路径（lambert/
three_body_lambert/multi_impulse）、自然动力学路径（low_energy/manifold，
覆盖引力辅助数学内核）、低推力路径（low_thrust/）、任务层（search/
optimize/porkchop）。``transfer_orbit.py`` 是编排器：接收 transfer_type
（HMN/LGA/WSB/小推力），按枚举选路径组合底层数学模块（新类型，当前占位）。

未实现（对外承诺能力）：LGA/WSB 引力辅助弹道搜索，占位抛
``NotImplementedError``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import (
    TransferArc,
    TransferConfig,
    TransferOptimizationResult,
    TransferSolution,
)
from .lambert import LambertSolution, solve_lambert, solve_lambert_batch
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
    "qlaw_guess",
    "rv_to_keplerian",
]


@dataclass
class TransferDesignResult:
    """转移轨道设计结果。

    Attributes:
        transfer_type: 转移类型（"HMN"/"LGA"/"WSB"/"low_thrust"）。
        delta_v: 总 Δv（km/s）。
        trajectory: 转移轨迹。
        details: 设计细节（弹道参数汇总）。
    """

    transfer_type: str
    delta_v: float
    trajectory: Any
    details: dict[str, Any] = field(default_factory=dict)


def transfer_orbit(
    transfer_type: str,
    *,
    target_ephemeris: Any = None,
    tli_params: Any = None,
    tof_range: tuple[float, float] | None = None,
    **kwargs,
) -> TransferDesignResult:
    """端到端转移轨道设计（编排器）。

    Args:
        transfer_type: "HMN"（直接）/ "LGA"（月球引力辅助）/ "WSB"（太阳引力辅助）/
            "low_thrust"（小推力）。
        target_ephemeris: 目标轨道星历（FR1 产物）。
        tli_params: 地球停泊轨道参数（TLI 高度/倾角/航迹角）。
        tof_range: 飞行时间范围（天）。

    Raises:
        NotImplementedError: 编排器实现未完成（当前为占位，能力在规划中）。
    """
    raise NotImplementedError(
        f"transfer_orbit('{transfer_type}') 实现未完成（能力在规划中）"
    )
