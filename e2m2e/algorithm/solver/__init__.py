"""迭代求解器薄封装（问题构造入口）。

下沉 Rust 的算法（ADR 0011）：``DifferentialCorrection``/``Continuation``/
``MultipleShooting``/``TwoLevelMultipleShooting`` 类名保留，Python 侧是
"问题构造入口"（约束/自由变量/目标配置），迭代循环/收敛判断最终在 Rust
``solver/``。本期只迁移文件位置 + 保持现有实现（源：``algorithms/`` 的
continuation/differential_correction/multiple_shooting/
two_level_multiple_shooting），下沉 Rust 是后续独立工作。
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
from .two_level_multiple_shooting import (
    TwoLevelMultipleShooting,
    TwoLevelMultipleShootingResult,
)

__all__ = [
    "DifferentialCorrection",
    "Continuation",
    "MultipleShooting",
    "MultipleShootingResult",
    "TwoLevelMultipleShooting",
    "TwoLevelMultipleShootingResult",
    "sample_patch_points",
    "sample_patch_points_perilune_clustered",
    "convert_to_j2000",
]
