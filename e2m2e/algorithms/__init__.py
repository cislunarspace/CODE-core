"""
e2m2e算法模块

包含用于轨道设计和优化的各种算法。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ephemeris_correction import EphemerisCorrectionResult as EphemerisCorrectionResult
    from .multiple_shooting import MultipleShooting as MultipleShooting
    from .two_level_multiple_shooting import (
        TwoLevelMultipleShooting as TwoLevelMultipleShooting,
        TwoLevelMultipleShootingResult as TwoLevelMultipleShootingResult,
    )

from . import continuation, differential_correction, halo_initial_guess, stability, strategies
from .continuation import Continuation
from .differential_correction import DifferentialCorrection
from .halo_initial_guess import (
    compute_halo_coefficients,
    compute_halo_initial_guess,
    halo_third_order_approximation,
)
from .stability import BifurcationType, StabilityAnalysis, StabilityType
from .strategies import CorrectionConfig

# 星历相关算法模块通过 __getattr__ 按需延迟导入，避免用户只使用 CR3BP 算法时
# 强制加载 spiceypy。参见 issue #44。
_LAZY_MODULE_EXPORTS: dict[str, str] = {
    "ephemeris_correction": "e2m2e.algorithms.ephemeris_correction",
    "multiple_shooting": "e2m2e.algorithms.multiple_shooting",
    "two_level_multiple_shooting": "e2m2e.algorithms.two_level_multiple_shooting",
}

_LAZY_SYMBOL_EXPORTS: dict[str, str] = {
    "EphemerisCorrectionResult": "e2m2e.algorithms.ephemeris_correction",
    "correct_ephemeris_patch_points": "e2m2e.algorithms.ephemeris_correction",
    "MultipleShooting": "e2m2e.algorithms.multiple_shooting",
    "convert_to_j2000": "e2m2e.algorithms.multiple_shooting",
    "sample_patch_points": "e2m2e.algorithms.multiple_shooting",
    "TwoLevelMultipleShooting": "e2m2e.algorithms.two_level_multiple_shooting",
    "TwoLevelMultipleShootingResult": "e2m2e.algorithms.two_level_multiple_shooting",
}


def __getattr__(name: str) -> object:
    """按需延迟导入星历相关算法模块及其公开符号。"""
    module_name = _LAZY_MODULE_EXPORTS.get(name)
    if module_name is not None:
        module = importlib.import_module(module_name)
        globals()[name] = module
        return module

    module_name = _LAZY_SYMBOL_EXPORTS.get(name)
    if module_name is not None:
        module = importlib.import_module(module_name)
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'e2m2e.algorithms' has no attribute '{name}'")


def __dir__() -> list[str]:
    """确保 dir(e2m2e.algorithms) 包含延迟导出的公开符号。"""
    return sorted(
        set(__all__)
        | set(globals().keys())
        | set(_LAZY_MODULE_EXPORTS.keys())
        | set(_LAZY_SYMBOL_EXPORTS.keys())
    )


__all__ = [
    "differential_correction",
    "continuation",
    "stability",
    "ephemeris_correction",
    "multiple_shooting",
    "two_level_multiple_shooting",
    "strategies",
    "halo_initial_guess",
    "compute_halo_coefficients",
    "halo_third_order_approximation",
    "compute_halo_initial_guess",
    "DifferentialCorrection",
    "Continuation",
    "StabilityAnalysis",
    "StabilityType",
    "BifurcationType",
    "MultipleShooting",
    "EphemerisCorrectionResult",
    "correct_ephemeris_patch_points",
    "TwoLevelMultipleShooting",
    "TwoLevelMultipleShootingResult",
    "sample_patch_points",
    "convert_to_j2000",
    "CorrectionConfig",
]
