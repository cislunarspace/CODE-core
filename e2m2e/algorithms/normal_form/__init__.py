"""标准形（Normal Form）算法包。

为圆型限制性三体问题（CR3BP）平动点附近的轨道设计提供标准形化简流水线
的基础脚手架：

- 切片 0 交付 ``NormalFormContext``、``NormalFormResult`` 与 SI ↔ qiao
  归一化单位转换缝；
- 切片 1 交付 Legendre 展开、符号 Hamilton 量构造与数值 Hamilton 量
  时间序列；
- 切片 2（本切片）交付动力学替代 corrector、生成函数 ``W``、FFT
  频率分析、块三对角多重打靶法。

后续切片在此基础上构建化简器（``QuasiFloquetReducer``、
``CenterManifoldReducer`` 等）。

归一化约定沿用 qiao ``Global_File.py``：长度单位 LU（km）、
时间单位 TU（s）、速度单位 VU = LU/TU（km/s）。本包不强制依赖
``sympy`` / ``joblib``：它们仅在 Legendre / Hamiltonian 模块内部
惰性导入，因此 ``import e2m2e.algorithms.normal_form`` 不会触发
重依赖加载。NAFF 是可选依赖：``fft`` 模块默认走 FFT 回退，仅在
显式调用 :func:`detect_naff` 时才会查找外部可执行文件。
"""

from __future__ import annotations

from .context import NormalFormContext
from .types import NormalFormResult

__all__ = [
    "NormalFormContext",
    "NormalFormResult",
]

# 切片 2 公共符号从子模块惰性导入；放在 ``__all__`` 末尾以便
# 用户 ``from e2m2e.algorithms.normal_form import DynamicalSubstituteCorrector``
# 直接工作，同时不强制 ``import e2m2e.algorithms.normal_form`` 在无
# scipy / 无 Orbit 的环境下也成功。
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
    # 切片 3（quasi-Floquet 变换矩阵 B(t)）
    "QuasiFloquetReducer": "quasi_floquet",
    "QuasiFloquetResult": "quasi_floquet",
    # 切片 4（高阶中心流形化简，Code10–Code11）
    "CenterManifoldReducer": "center_manifold",
    "CenterManifoldResult": "center_manifold",
    # 切片 5（坐标变换链 rho↔param，issue #174）
    "LibrationCatalogData": "catalog",
    "LibrationCatalogTransformer": "catalog",
    # 切片 6（一键式流水线，issue #175）
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
