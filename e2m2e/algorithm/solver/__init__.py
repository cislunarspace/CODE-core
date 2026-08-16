"""迭代求解器的 Python 问题构造与编排入口。

ADR 0011 的下沉边界：``DifferentialCorrection`` 的 CR3BP 数值内核已经在
Rust，Python 侧保留对称性配置、问题构造、结果与 Orbit 编排；``Continuation``
和 ``MultipleShooting`` 类仍保留 Python 编排或数值实现，按各自迁移工作项推进。

星历 patch points 修正的旧 Python 求解器（``MultipleShooting``/`
``TwoLevelMultipleShooting`` 包装层 ``ephemeris_correction`` 子包）已删除：
设计链路统一走 Rust ``multiple_shooting_correct_py``（segmented 与稳定轨道
默认路径）。``MultipleShooting`` 本身保留（transfer/hohmann 等仍使用）。
"""

from __future__ import annotations

from .continuation import Continuation
from .differential_correction import DifferentialCorrection
from .multiple_shooting import (
    MultipleShooting,
    MultipleShootingResult,
    convert_to_j2000,
    sample_patch_points,
    sample_patch_points_perilune_clustered,
)

__all__ = [
    "DifferentialCorrection",
    "Continuation",
    "MultipleShooting",
    "MultipleShootingResult",
    "sample_patch_points",
    "sample_patch_points_perilune_clustered",
    "convert_to_j2000",
]
