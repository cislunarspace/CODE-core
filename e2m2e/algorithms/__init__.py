"""
e2m2e算法模块

包含用于轨道设计和优化的各种算法。
"""

from . import continuation, differential_correction, multiple_shooting, stability, strategies
from .continuation import Continuation
from .differential_correction import (
    DifferentialCorrection,
    compute_halo_coefficients,
    compute_halo_initial_guess,
    halo_third_order_approximation,
)
from .multiple_shooting import MultipleShooting, convert_to_j2000, sample_patch_points
from .stability import BifurcationType, StabilityAnalysis, StabilityType
from .strategies import CorrectionConfig

__all__ = [
    "differential_correction",
    "continuation",
    "stability",
    "multiple_shooting",
    "strategies",
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
    "CorrectionConfig",
]
