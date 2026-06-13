"""e2m2e 转移轨道设计模块

提供 DRO-RO 转移轨道的网格搜索和 NLP 优化工具，实现 Cui et al. (2025) 的"搜索-优化"两步法。

Classes:
    SearchConfig: 搜索/优化参数配置 dataclass
    TransferSearch: 通用轨道转移网格搜索
    Transfer: 简化的转移轨道优化接口
    TransferConfig: 转移优化配置
    TransferOptimizationResult: 转移优化结果
    DROTRONLPOptimizer: DRO-RO 转移轨道 NLP 优化器
    NLPOptimizationVariables: NLP 优化变量

Functions:
    load_orbit_from_json: 从 JSON 文件加载轨道数据
    optimize_transfer: 便捷函数：优化 DRO 到 RO 转移
    optimize_with_copt: 使用 COPT 求解 NLP
"""

from .propulsion import ImpulsivePropulsion, PropulsionModel
from .terminal import OrbitTerminal, StateTerminal, TerminalCondition
from .config import TransferConfig, TransferOptimizationResult
from .search_config import SearchConfig
from .transfer import Transfer
from .transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
    optimize_transfer,
    optimize_with_copt,
)
from .transfer_search import (
    DEFAULT_MIN_DISTANCE_THRESHOLD_DU,
    DROROTransferSearch,
    DROTransferSearch,
    TransferSearch,
    load_orbit_from_json,
)

_HAVE_COPT = transfer_optimization.coptpy is not None

__all__ = [
    "SearchConfig",
    "TransferSearch",
    "DROTransferSearch",
    "DROROTransferSearch",
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
