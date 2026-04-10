"""e2m2e 转移轨道设计模块

提供 DRO-RO 转移轨道的网格搜索和 NLP 优化工具，实现 Cui et al. (2025) 的"搜索-优化"两步法。

Classes:
    TransferSearch: 通用轨道转移网格搜索
    Transfer: 简化的转移轨道优化接口
    DROTRONLPOptimizer: DRO-RO 转移轨道 NLP 优化器
    COPTNLPSolver: 基于 COPT 的 NLP 求解器封装

Functions:
    load_orbit_from_json: 从 JSON 文件加载轨道数据
    optimize_transfer: 便捷函数：优化 DRO 到 RO 转移
    optimize_with_copt: 使用 COPT 求解 NLP
"""

from . import transfer_search
from . import transfer_optimization
from . import transfer

from .transfer_search import (
    TransferSearch,
    DROTransferSearch,
    DROROTransferSearch,
    load_orbit_from_json,
    DEFAULT_MIN_DISTANCE_THRESHOLD_DU,
)

from .transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
    NLPOptimizationResult,
    TransferType,
    optimize_transfer,
    optimize_with_copt,
)

from .transfer import (
    Transfer,
    TransferConfig,
    TransferOptimizationResult,
)

_HAVE_COPT = transfer_optimization.coptpy is not None

__all__ = [
    "TransferSearch",
    "DROTransferSearch",
    "DROROTransferSearch",
    "Transfer",
    "TransferConfig",
    "TransferOptimizationResult",
    "DROTRONLPOptimizer",
    "NLPOptimizationResult",
    "NLPOptimizationVariables",
    "TransferType",
    "DEFAULT_MIN_DISTANCE_THRESHOLD_DU",
    "load_orbit_from_json",
    "optimize_transfer",
    "optimize_with_copt",
    "_HAVE_COPT",
]
