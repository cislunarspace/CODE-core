"""任务轨道设计（三段编排）。

回答"一个任务参数怎么变成一条可用的标称轨道"。三段编排（ADR 0011 迁移，
源：``dfh/design_orbit.py``）：family（初猜）→ 星历修正（Rust 多重打靶
``multiple_shooting_correct_py``，segmented 或稳定轨道默认路径）→
propagation（高精度预报）。六类轨道：
DRO / NRHO / Halo / Lissajous / L4 / L5。api/ 只做 Pydantic 校验 +
薄调用，编排逻辑留这里。
"""

from __future__ import annotations

from .design_orbit import (
    DEFAULT_DESIGN_PERTURBATION,
    DesignNotConvergedError,
    OrbitDesignResult,
    default_kernel_dir,
    design_orbit,
    load_design_kernels,
)

__all__ = [
    "OrbitDesignResult",
    "design_orbit",
    "DesignNotConvergedError",
    "default_kernel_dir",
    "load_design_kernels",
    "DEFAULT_DESIGN_PERTURBATION",
]
