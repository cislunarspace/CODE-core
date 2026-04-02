"""
e2m2e算法模块

包含用于轨道设计和优化的各种算法。
"""

from . import differential_correction
from . import continuation
from . import stability
from . import multiple_shooting

from .differential_correction import (
    DifferentialCorrection,
    compute_halo_coefficients,
    halo_third_order_approximation,
    compute_halo_initial_guess,
)
from .continuation import Continuation, ContinuationMethod
from .stability import StabilityAnalysis
from .multiple_shooting import MultipleShooting

__all__ = [
    "differential_correction",
    "continuation",
    "stability",
    "multiple_shooting",
    "compute_halo_coefficients",
    "halo_third_order_approximation",
    "compute_halo_initial_guess",
    "DifferentialCorrection",
    "Continuation",
    "ContinuationMethod",
    "StabilityAnalysis",
    "MultipleShooting",
]
