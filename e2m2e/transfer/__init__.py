"""
e2m2e转移轨道设计模块

提供轨道转移轨道的搜索和优化工具。

使用方式:
    from e2m2e.transfer import TransferSearch

    results = TransferSearch(system, dynamics).search(
        alpha_min=0.5,
        alpha_max=2.5,
        n_alpha=101,
        n_departure=200,
        max_transfer_time=15.0,
        intersection_threshold=0.001,
        min_distance_threshold=0.001,
        collision_earth_radius=0.0005,
        collision_moon_radius=0.0003,
        integration_dt=0.01,
        departure_orbit=departure_orbit,
        arrival_orbit=arrival_orbit,
    )
"""

from . import transfer_base
from . import transfer_search
from . import transfer_optimization

from .transfer_search import (
    TransferSearch,
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
    "TransferSearch",
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
