"""e2m2e 转移轨道设计模块

提供 DRO-RO 转移轨道的网格搜索和 NLP 优化工具，实现 Cui et al. (2025) 的"搜索-优化"两步法。
"""

from . import transfer_optimization
from .config import TransferConfig, TransferOptimizationResult
from .propulsion import ImpulsivePropulsion, PropulsionModel
from .search_config import SearchConfig  # 向后兼容别名（= TransferConfig）
from .terminal import OrbitTerminal, StateTerminal, TerminalCondition
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
    "SearchConfig",
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
    "PropulsionModel",
    "ImpulsivePropulsion",
    "TerminalCondition",
    "OrbitTerminal",
    "StateTerminal",
    "_HAVE_COPT",
]
