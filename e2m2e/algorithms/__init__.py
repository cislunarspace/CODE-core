"""
e2m2e算法模块

包含用于轨道设计和优化的各种算法。
"""

from . import analytical
from . import differential_correction
from . import continuation
from . import stability
from . import halo_orbit

from .analytical import (
    compute_halo_coefficients,
    halo_third_order_approximation,
    compute_halo_initial_guess,
)
from .differential_correction import DifferentialCorrection
from .continuation import Continuation, ContinuationMethod
from .stability import StabilityAnalysis
from .halo_orbit import HaloOrbitGenerator

__all__ = [
    "analytical",
    "differential_correction",
    "continuation",
    "stability",
    "halo_orbit",
    "compute_halo_coefficients",
    "halo_third_order_approximation",
    "compute_halo_initial_guess",
    "DifferentialCorrection",
    "Continuation",
    "ContinuationMethod",
    "StabilityAnalysis",
    "HaloOrbitGenerator",
]
