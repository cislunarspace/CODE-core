"""
e2m2e转移轨道设计模块

提供DRO到RO平面转移轨道的搜索和优化工具。

使用方式:
    from e2m2e.transfer import DROTransferSearch

    transfer = DROTransferSearch(system, dynamics)
    transfer.set_departure_orbit(dro_orbit)
    transfer.set_arrival_orbit(ro_orbit)
    transfer.alpha_min = 0.5
    transfer.alpha_max = 2.5
    # ... 设置其他参数
    results = transfer.search()
"""

from . import transfer_base
from . import transfer_search
from . import transfer_optimization

from .transfer_search import (
    DROTransferSearch,
    DROROTransferSearch,
    load_orbit_from_json,
    save_search_results,
)

from .transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
    NLPOptimizationResult,
    TransferType,
    optimize_transfer,
    optimize_with_copt,
)

from .transfer_base import (
    BaseTransfer,
    DEFAULT_MIN_DISTANCE_THRESHOLD_DU,
    TransferStrategy,
)

_HAVE_COPT = transfer_optimization.coptpy is not None

__all__ = [
    # 转移设计类
    "DROTransferSearch",
    "DROROTransferSearch",
    # NLP优化类
    "DROTRONLPOptimizer",
    # 结果类
    "NLPOptimizationResult",
    # 变量类
    "NLPOptimizationVariables",
    # 枚举
    "TransferType",
    "TransferStrategy",
    # 基类
    "BaseTransfer",
    "DEFAULT_MIN_DISTANCE_THRESHOLD_DU",
    # 工具函数
    "load_orbit_from_json",
    "save_search_results",
    "optimize_transfer",
    "optimize_with_copt",
    # 元信息
    "_HAVE_COPT",
]
