"""星历修正子包。

通过 ``PatchPointCorrector`` 注册表分发修正方法，
取代原先的 ``if/elif`` 字符串分发。

每种修正方法由一个私有 ``PatchPointCorrector`` 实现包装对应求解器，
将统一参数翻译为求解器特定参数，并将求解器结果重包为
``EphemerisCorrectionResult``。

子包结构：

- ``standard``  —— 包装 ``MultipleShooting``
- ``two_level`` —— 包装 ``TwoLevelMultipleShooting``
- ``homotopy``  —— 同伦过渡修正（``correct_with_homotopy``）

同伦修正与分发器同置于本子包内，因此分发器可正常 import
``correct_with_homotopy``，不再需要延迟导入的循环依赖 workaround。
接缝类型（``EphemerisCorrectionResult`` 等）保留在
``e2m2e.algorithms.ephemeris_correction_types`` 作为叶子模块破环。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..ephemeris_correction_types import (
    EphemerisCorrectionResult,
    PatchPointCorrector,
    UnsupportedCorrectorMethodError,
)
from .homotopy import _HomotopyPatchPointCorrector
from .standard import _StandardPatchPointCorrector
from .two_level import _TwoLevelPatchPointCorrector

# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., PatchPointCorrector]] = {
    "standard": lambda dynamics, **kw: _StandardPatchPointCorrector(dynamics, **kw),
    "two_level": lambda dynamics, **kw: _TwoLevelPatchPointCorrector(dynamics, **kw),
    "homotopy": lambda dynamics, **kw: _HomotopyPatchPointCorrector(dynamics, **kw),
}


# ---------------------------------------------------------------------------
# 公开分发函数（保持向后兼容）
# ---------------------------------------------------------------------------


def correct_ephemeris_patch_points(
    method: str,
    dynamics: Any,
    t_patch: np.ndarray,
    state_patch: np.ndarray,
    *,
    tolerance: float,
    max_iter: int,
    verbose: bool,
    n_workers: int,
    kernel_dir: str,
    velocity_tolerance: float | None = None,
    base_bodies: list[str] | None = None,
    lambda_steps: list[float] | None = None,
    inner_method: str = "standard",
) -> EphemerisCorrectionResult:
    """通过注册表分发星历 patch points 修正。

    Args:
        method: 修正方法名称（``"standard"``、``"two_level"``、``"homotopy"``）
        dynamics: 动力学模型对象
        t_patch: 时间节点数组
        state_patch: 状态量数组，形状 ``(N, 6)``
        tolerance: 位置残差收敛容差
        max_iter: 最大迭代次数
        verbose: 是否显示进度条
        n_workers: 并行工作进程/线程数
        kernel_dir: SPICE 内核目录路径
        velocity_tolerance: 速度残差容差（仅 ``"two_level"`` 使用）
        base_bodies: 同伦基础天体列表（仅 ``"homotopy"`` 使用）
        lambda_steps: 同伦 lambda 步长序列（仅 ``"homotopy"`` 使用）
        inner_method: 同伦内部修正方法（仅 ``"homotopy"`` 使用）

    Returns:
        ``EphemerisCorrectionResult``

    Raises:
        UnsupportedCorrectorMethodError: 未知的修正方法名称
    """
    factory = _REGISTRY.get(method)
    if factory is None:
        raise UnsupportedCorrectorMethodError(method, list(_REGISTRY))
    corrector = factory(
        dynamics,
        n_workers=n_workers,
        kernel_dir=kernel_dir,
        base_bodies=base_bodies,
        lambda_steps=lambda_steps,
        inner_method=inner_method,
    )
    return corrector.correct(
        t_patch,
        state_patch,
        max_iter=max_iter,
        tolerance=tolerance,
        velocity_tolerance=velocity_tolerance,
        verbose=verbose,
    )


__all__ = [
    "EphemerisCorrectionResult",
    "PatchPointCorrector",
    "UnsupportedCorrectorMethodError",
    "correct_ephemeris_patch_points",
    "_REGISTRY",
    "_StandardPatchPointCorrector",
    "_TwoLevelPatchPointCorrector",
    "_HomotopyPatchPointCorrector",
]
