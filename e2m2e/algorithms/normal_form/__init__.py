"""标准形算法包 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.normal_form``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.normal_form import (
    NormalFormContext,
    NormalFormResult,
)
from e2m2e.algorithm.normal_form import (
    __getattr__ as _lazy_getattr,
)

__all__ = ["NormalFormContext", "NormalFormResult"]

# 旧路径的惰性导出委托给新模块（PEP 562）。
_LAZY_EXPORTS = {
    "DynamicalSubstituteCorrector": "dynamical_substitution",
    "DynamicalSubstituteResult": "dynamical_substitution",
    "FFTComponent": "fft",
    "extract_frequencies": "fft",
    "fft_extract": "fft",
    "frequency_match": "fft",
    "least_squares_sin_cos_fit": "fft",
    "naff_available": "fft",
    "reconstruct_signal": "fft",
    "reconstruct_derivative": "fft",
    "ODESubstituteSolver": "multiple_shooting",
    "ShootingPatch": "multiple_shooting",
    "MultipleShootingResult": "multiple_shooting",
    "multiple_shooting_newton": "multiple_shooting",
    "solve_block_tridiagonal": "multiple_shooting",
    "SubstituteSolver": "multiple_shooting",
    "QuasiFloquetReducer": "quasi_floquet",
    "QuasiFloquetResult": "quasi_floquet",
    "CenterManifoldReducer": "center_manifold",
    "CenterManifoldResult": "center_manifold",
    "LibrationCatalogData": "catalog",
    "LibrationCatalogTransformer": "catalog",
    "NormalFormPipeline": "pipeline",
}


def __getattr__(name: str):  # PEP 562
    if name in _LAZY_EXPORTS:
        value = _lazy_getattr(name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


for _name in _LAZY_EXPORTS:
    __all__.append(_name)
