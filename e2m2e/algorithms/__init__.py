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
from .continuation import Continuation
from .stability import StabilityAnalysis, StabilityType, BifurcationType
from .multiple_shooting import MultipleShooting
from .multiple_shooting import sample_patch_points, convert_to_j2000

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
    "StabilityAnalysis",
    "StabilityType",
    "BifurcationType",
    "MultipleShooting",
    "sample_patch_points",
    "convert_to_j2000",
]
