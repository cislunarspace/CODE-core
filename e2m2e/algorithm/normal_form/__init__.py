"""标准形（Normal Form）算法包（ADR 0011 迁移，源：``algorithms/normal_form/``）。

为圆型限制性三体问题（CR3BP）平动点附近的轨道设计提供标准形化简流水线
的基础脚手架（算法层子模块 + 可选依赖 ``[normal-form]`` + 惰性导入 +
不注册 MCP，三档辅助）。

归一化约定沿用 qiao ``Global_File.py``：长度单位 LU（km）、时间单位
TU（s）、速度单位 VU = LU/TU（km/s）。本包不强制依赖 ``sympy`` /
``joblib``：它们仅在 Legendre / Hamiltonian 模块内部惰性导入。
"""

from __future__ import annotations

from .context import NormalFormContext
from .types import NormalFormResult

__all__ = [
    "NormalFormContext",
    "NormalFormResult",
]

# 公共符号从子模块惰性导出，避免顶层导入强制加载 sympy 等重依赖。
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
        from importlib import import_module

        module = import_module(f"{__name__}.{_LAZY_EXPORTS[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


for _name in _LAZY_EXPORTS:
    __all__.append(_name)
