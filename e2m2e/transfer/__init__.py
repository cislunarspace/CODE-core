"""
e2m2e转移轨道设计模块

提供轨道转移轨道的搜索和优化工具。

使用方式:
    from e2m2e.transfer import Transfer

    transfer = Transfer(dynamics)
    transfer.set_orbit(start=dro_orbit, end=ro_orbit)
    result = transfer.optimize(
        initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
        alpha_range=(0.5, 2.5),
    )
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
