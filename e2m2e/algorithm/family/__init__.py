"""轨道族生成：种子/初猜/族行走/注册表。

回答"一条轨道/一族轨道怎么收敛出来"（ADR 0011）。六类初猜函数（design_dro/
design_halo/design_nrho/design_lissajous/design_triangular）待从
``dfh/cr3bp_orbits.py`` 拆入；轨道族注册表 = 函数形态（``_REGISTRY: dict[str,
Callable]``），新族 = 写一个设计函数 + 注册。

实现状态：骨架。六类初猜函数待迁入。
"""

from __future__ import annotations

from collections.abc import Callable

from ...data.types import Orbit

__all__ = [
    "design_dro",
    "design_halo",
    "design_nrho",
    "design_lissajous",
    "design_triangular",
    "registry",
]


#: 轨道族注册表（函数形态）：orbit_type → design_xxx(params) -> Orbit。
registry: dict[str, Callable[..., Orbit]] = {}


def design_dro(amplitude_km: float, **kwargs) -> Orbit:
    """生成指定振幅的 DRO 周期轨道。

    实现状态：待迁入（源 dfh/cr3bp_orbits.py）。
    """
    raise NotImplementedError("design_dro 待从 dfh/cr3bp_orbits.py 迁入")


def design_halo(collinear_point: int, amplitude_km: float, **kwargs) -> Orbit:
    """生成指定面外振幅的 Halo 周期轨道（正北负南）。

    实现状态：待迁入（源 dfh/cr3bp_orbits.py）。
    """
    raise NotImplementedError("design_halo 待从 dfh/cr3bp_orbits.py 迁入")


def design_nrho(
    collinear_point: int, north_south: int, perilune_height_km: float, **kwargs
) -> Orbit:
    """生成指定近月点高度的 NRHO 周期轨道。

    实现状态：待迁入（源 dfh/cr3bp_orbits.py）。
    """
    raise NotImplementedError("design_nrho 待从 dfh/cr3bp_orbits.py 迁入")


def design_lissajous(
    collinear_point: int, amplitude_in_km: float, amplitude_out_km: float,
    phase_in: float, phase_out: float, **kwargs,
) -> Orbit:
    """生成指定共线点的 Lissajous 拟周期轨道初猜。

    实现状态：待迁入（源 dfh/cr3bp_orbits.py）。
    """
    raise NotImplementedError("design_lissajous 待从 dfh/cr3bp_orbits.py 迁入")


def design_triangular(
    point: int, amplitude_in_km: float, amplitude_out_km: float,
    phase_in: float, phase_out: float, **kwargs,
) -> Orbit:
    """生成 L4/L5 邻域拟周期轨道初猜。

    实现状态：待迁入（源 dfh/cr3bp_orbits.py）。
    """
    raise NotImplementedError("design_triangular 待从 dfh/cr3bp_orbits.py 迁入")
