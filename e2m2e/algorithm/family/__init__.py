"""轨道族生成：种子/初猜/族行走/注册表。

回答"一条轨道/一族轨道怎么收敛出来"（ADR 0011 迁移，源：
``dfh/cr3bp_orbits.py`` 六类初猜 + ``algorithms/halo_family.py`` +
``algorithms/halo_initial_guess.py`` + ``algorithms/lissajous_initial_guess.py`` +
``algorithms/triangular_initial_guess.py`` + ``algorithms/strategies/``）。

轨道族注册表 = 函数形态（``registry: dict[str, Callable]``），注册表值 =
设计函数 ``design_xxx(params) -> Orbit``；``algorithm/design`` 的
``design_orbit`` 查注册表按族分发（新族 = 写一个设计函数 + 注册）。

``cr3bp_orbits``（六类初猜）依赖 ``algorithm/solver``，而 solver 依赖本包
的 ``halo_initial_guess``——为打破包级循环导入，初猜函数经 PEP 562
``__getattr__`` 惰性导出。
"""

from __future__ import annotations

from collections.abc import Callable

from ...data.types import Orbit

# 先加载 solver 依赖的叶子模块（halo_initial_guess 等），再经惰性导出
# cr3bp_orbits（依赖 solver）。
from .axial_initial_guess import compute_axial_initial_guess
from .halo_family import (
    generate_halo_family,
    generate_halo_seed_orbit,
    halo_pseudo_arclength_continuation,
)
from .halo_initial_guess import (
    compute_halo_coefficients,
    compute_halo_initial_guess,
    halo_third_order_approximation,
)
from .lissajous_initial_guess import compute_lissajous_initial_guess
from .strategies import (
    CorrectionConfig,
    axial_fixed_vz0,
    halo_fixed_x0,
    halo_fixed_z0,
    symmetric_2d_fixed_t,
    symmetric_2d_fixed_x0,
    symmetric_2d_fixed_y0,
    symmetric_3d_fixed_x0,
    symmetric_xz_fixed_x0,
    symmetric_xz_fixed_z0,
)
from .triangular_initial_guess import compute_triangular_initial_guess

__all__ = [
    "design_axial",
    "design_dro",
    "design_halo",
    "design_nrho",
    "design_lissajous",
    "design_triangular",
    "earth_moon_system",
    "Cr3bpOrbitError",
    "compute_axial_initial_guess",
    "compute_halo_initial_guess",
    "compute_halo_coefficients",
    "halo_third_order_approximation",
    "compute_lissajous_initial_guess",
    "compute_triangular_initial_guess",
    "generate_halo_seed_orbit",
    "generate_halo_family",
    "halo_pseudo_arclength_continuation",
    "CorrectionConfig",
    "symmetric_2d_fixed_x0",
    "symmetric_2d_fixed_t",
    "symmetric_2d_fixed_y0",
    "symmetric_3d_fixed_x0",
    "symmetric_xz_fixed_x0",
    "symmetric_xz_fixed_z0",
    "halo_fixed_z0",
    "halo_fixed_x0",
    "axial_fixed_vz0",
    "registry",
]

#: 惰性导出：cr3bp_orbits（六类初猜 + earth_moon_system + Cr3bpOrbitError）
#: 依赖 ``algorithm/solver``，经 PEP 562 在首次访问时加载。
_LAZY_EXPORTS = {
    "design_axial": "design_axial",
    "design_dro": "design_dro",
    "design_halo": "design_halo",
    "design_nrho": "design_nrho",
    "design_lissajous": "design_lissajous",
    "design_triangular": "design_triangular",
    "earth_moon_system": "earth_moon_system",
    "Cr3bpOrbitError": "Cr3bpOrbitError",
}


def __getattr__(name: str):  # PEP 562
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module = import_module(f"{__name__}.cr3bp_orbits")
        value = getattr(module, _LAZY_EXPORTS[name])
        globals()[name] = value
        return value
    if name == "registry":
        value = _build_registry()
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _build_registry() -> dict[str, Callable[..., Orbit]]:
    """构建轨道族注册表（函数形态）：orbit_type → design_xxx(params) -> Orbit。"""
    from .cr3bp_orbits import (
        design_axial,
        design_dro,
        design_halo,
        design_lissajous,
        design_nrho,
        design_triangular,
    )

    return {
        "DRO": design_dro,
        "HALO": design_halo,
        "NRHO": design_nrho,
        "LISSAJOUS": design_lissajous,
        "AXIAL": design_axial,
        "L4": lambda amplitude_in, amplitude_out, phase_in=0.0, phase_out=0.0, **kw: (
            design_triangular(4, amplitude_in, amplitude_out, phase_in, phase_out, **kw)
        ),
        "L5": lambda amplitude_in, amplitude_out, phase_in=0.0, phase_out=0.0, **kw: (
            design_triangular(5, amplitude_in, amplitude_out, phase_in, phase_out, **kw)
        ),
    }
