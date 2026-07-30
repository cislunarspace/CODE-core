"""e2m2e 转移轨道设计模块

提供 DRO-RO 转移轨道的网格搜索和 NLP 优化工具，实现 Cui et al. (2025) 的"搜索-优化"两步法。
"""

from . import transfer_optimization
from .config import (
    TransferArc,
    TransferConfig,
    TransferOptimizationResult,
    TransferSolution,
)
from .lambert import LambertSolution, solve_lambert, solve_lambert_batch
from .low_energy import PatchCandidate, design_low_energy_transfer, patch_manifolds
from .lowthrust_shooting import (
    EngineConfig,
    LowThrustSegment,
    LowThrustShooting,
    LowThrustShootingSolution,
)
from .multi_impulse import (
    CoastArc,
    Impulse,
    MultiImpulseTransfer,
    PrimerVectorReport,
)
from .porkchop import PorkchopData, porkchop
from .propulsion import ImpulsivePropulsion
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

_HAVE_COPT = transfer_optimization.coptpy is not None

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
    "_HAVE_COPT",
]
