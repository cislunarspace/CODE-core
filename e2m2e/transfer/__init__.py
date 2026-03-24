"""
e2m2e转移轨道设计模块

提供地球到月球、月球到地球以及轨道间转移的设计工具。

使用方式:
    # 新的推荐方式
    from e2m2e.transfer import DROTransferSearch

    transfer = DROTransferSearch(system, dynamics)
    transfer.set_departure_orbit(dro_orbit)
    transfer.set_arrival_orbit(ro_orbit)
    transfer.configure_search(alpha_range=(0.5, 2.5))
    results = transfer.search()

    # 传统方式 (保持向后兼容)
    from e2m2e.transfer import DROROTransferSearch, TransferSearchConfig

    config = TransferSearchConfig()
    searcher = DROROTransferSearch(system, dynamics)
    results = searcher.grid_search(departure_orbit, arrival_orbit)
"""

from . import earth_moon_transfer
from . import moon_earth_transfer
from . import orbit_to_orbit_transfer
from . import dro_transfer_optimization
from . import transfer_base

from .earth_moon_transfer import EarthMoonTransfer
from .moon_earth_transfer import MoonEarthTransfer
from .orbit_to_orbit_transfer import InterOrbitTransfer
from .dro_transfer_search import (
    DROTransferSearch,
    DROROTransferSearch,
    TransferSearchConfig,
    TransferSearchResult,
    load_orbit_from_json,
    save_search_results,
)
from .dro_transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
    NLPOptimizationResult,
    TransferType,
    optimize_transfer,
)

# 导出基础模块的类
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

__all__ = [
    # 转移设计类
    "EarthMoonTransfer",
    "MoonEarthTransfer",
    "InterOrbitTransfer",
    "DROTransferSearch",
    "DROROTransferSearch",
    "DROTRONLPOptimizer",
    # 配置类
    "TransferSearchConfig",
    "TransferConfig",
    "SearchConfig",
    "OptimizationConfig",
    # 结果类
    "TransferSearchResult",
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
]
