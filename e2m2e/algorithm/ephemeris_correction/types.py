"""星历修正接缝类型

定义 ``PatchPointCorrector`` 接缝与 ``EphemerisCorrectionResult`` 结果类型，
供 ``ephemeris_correction`` 子包的注册表与各修正实现共用。

作为子包内的叶子模块破环：分发器与各修正实现从 ``.types`` import，
避免分发器与实现之间的循环依赖。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ...data.templates import ConvergenceState, FailureCause
from ..results import ResultStatus


@dataclass(frozen=True)
class EphemerisCorrectionResult:
    """星历修正统一结果。

    所有 ``PatchPointCorrector`` 实现都必须返回此类型，
    消除各求解器结果类型不一致带来的重包逻辑。

    Attributes:
        converged: 是否收敛
        iterations: 迭代次数
        max_residual: 最终最大残差
        residual_history: 每次迭代最大残差的历史记录
        t_patch: 修正后的时间节点数组
        state_patch: 修正后的状态量数组
        velocity_residual: 速度残差（仅两层修正有值）
        velocity_residual_history: 速度残差历史（仅两层修正有值）
    """

    status: ConvergenceState
    cause: FailureCause
    message: str
    iterations: int
    max_residual: float
    residual_history: list[float]
    t_patch: np.ndarray
    state_patch: np.ndarray
    velocity_residual: float | None = None
    velocity_residual_history: list[float] | None = None

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@runtime_checkable
class PatchPointCorrector(Protocol):
    """星历 patch points 修正接缝。

    所有修正方法（standard、two_level、homotopy）都通过此接缝暴露。
    修正器特定的构造参数（``n_workers``、``kernel_dir``、``base_bodies`` 等）
    通过构造器注入；``correct`` 只接收统一的求解参数。
    """

    def correct(
        self,
        t_patch: np.ndarray,
        state_patch: np.ndarray,
        *,
        max_iter: int,
        tolerance: float,
        velocity_tolerance: float | None,
        verbose: bool,
    ) -> EphemerisCorrectionResult:
        """执行 patch points 修正并返回统一结果。

        Args:
            t_patch: 时间节点数组
            state_patch: 状态量数组，形状 ``(N, 6)``
            max_iter: 最大迭代次数
            tolerance: 位置残差收敛容差
            velocity_tolerance: 速度残差容差（不适用时为 ``None``）
            verbose: 是否显示进度条

        Returns:
            ``EphemerisCorrectionResult``
        """
        ...  # pragma: no cover


class UnsupportedCorrectorMethodError(ValueError):
    """注册表中找不到请求的修正方法时抛出。"""

    def __init__(self, method: str, available: Sequence[str]) -> None:
        self.method = method
        self.available = list(available)
        super().__init__(
            f"unsupported correction method: {method!r}; available: {', '.join(sorted(available))}"
        )
