"""
e2m2e转移轨道设计模块

提供DRO到RO平面转移轨道的搜索和优化工具。

使用方式:
    from e2m2e.transfer import DROTransferSearch

    transfer = DROTransferSearch(system, dynamics)
    transfer.set_departure_orbit(dro_orbit)
    transfer.set_arrival_orbit(ro_orbit)
    transfer.configure_search(alpha_range=(0.5, 2.5))
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
)

from .transfer_base import (
    BaseTransfer,
    TransferStrategy,
    TransferConfig,
    SearchConfig,
    OptimizationConfig,
    TransferResult,
    SearchResult,
    OptimizationResult,
)

_HAVE_COPT = transfer_optimization is not None

__all__ = [
    # 转移设计类
    "DROTransferSearch",
    "DROROTransferSearch",
    # NLP优化类
    "DROTRONLPOptimizer",
    # 配置类
    "TransferConfig",
    "SearchConfig",
    "OptimizationConfig",
    # 结果类
    "TransferResult",
    "SearchResult",
    "NLPOptimizationResult",
    # 变量类
    "NLPOptimizationVariables",
    # 枚举
    "TransferType",
    "TransferStrategy",
    # 基类
    "BaseTransfer",
    # 工具函数
    "load_orbit_from_json",
    "save_search_results",
    "optimize_transfer",
    # 元信息
    "_HAVE_COPT",
]
