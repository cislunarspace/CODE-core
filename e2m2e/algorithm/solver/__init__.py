"""迭代求解器薄封装（问题构造入口）。

下沉 Rust 的算法（ADR 0011）：``DifferentialCorrection``/``Continuation``/
``MultipleShooting`` 类名保留，Python 侧是"问题构造入口"（约束/自由变量/
目标配置），迭代循环/收敛判断最终在 Rust ``solver/``。本期只迁移文件位置 +
保持现有实现（源：``algorithms/`` 的 continuation/differential_correction/
multiple_shooting），下沉 Rust 是后续独立工作。

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
