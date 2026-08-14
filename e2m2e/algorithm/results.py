"""算法结果的统一最终状态契约。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import numpy as np

from ..data.templates import ConvergenceState, FailureCause

if TYPE_CHECKING:
    from ..data.types.orbit import Orbit, OrbitFamily


CandidateT = TypeVar("CandidateT")


CAUSE_STATUS: dict[FailureCause, ConvergenceState] = {
    FailureCause.NONE: ConvergenceState.CONVERGED,
    FailureCause.INTEGRATION_FAILED: ConvergenceState.FAILED,
    FailureCause.SINGULAR_JACOBIAN: ConvergenceState.FAILED,
    FailureCause.INVALID_PERIOD: ConvergenceState.INFEASIBLE,
    FailureCause.MAX_ITERATIONS_REACHED: ConvergenceState.MAX_ITERATIONS,
    FailureCause.STAGNATION_DETECTED: ConvergenceState.STAGNATED,
    FailureCause.DIVERGENCE_DETECTED: ConvergenceState.DIVERGED,
    FailureCause.NO_INTERSECTION: ConvergenceState.INFEASIBLE,
    FailureCause.CONSTRAINT_VIOLATION: ConvergenceState.INFEASIBLE,
    FailureCause.BODY_COLLISION: ConvergenceState.COLLISION,
    FailureCause.LEVEL1_CORRECTION_FAILED: ConvergenceState.FAILED,
    FailureCause.BACKEND_FAILURE: ConvergenceState.FAILED,
    FailureCause.INVALID_INPUT: ConvergenceState.FAILED,
    FailureCause.UNKNOWN: ConvergenceState.FAILED,
}


@dataclass(frozen=True)
class ResultStatus:
    """同步算法结果共享的最终状态三元组。"""

    status: ConvergenceState
    cause: FailureCause
    message: str

    def __post_init__(self) -> None:
        if self.status is ConvergenceState.ITERATING:
            raise ValueError("同步算法结果不能以 ITERATING 结束")
        expected = CAUSE_STATUS[self.cause]
        if self.status is not expected:
            raise ValueError(
                f"状态与原因不一致：{self.status.value} 不对应 {self.cause.value}；"
                f"应为 {expected.value}"
            )


@dataclass(frozen=True)
class CandidateSearchResult(Sequence[CandidateT], Generic[CandidateT]):
    """候选网格搜索的最终结果。

    保持序列读取接口，使既有调用方可继续迭代、索引和取长度；搜索是否
    完成及空结果的原因则由统一状态三元组明确表达。
    """

    candidates: tuple[CandidateT, ...]
    status: ConvergenceState
    cause: FailureCause
    message: str

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)

    def __len__(self) -> int:
        return len(self.candidates)

    @overload
    def __getitem__(self, index: int) -> CandidateT: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[CandidateT]: ...

    def __getitem__(self, index: int | slice) -> CandidateT | Sequence[CandidateT]:
        return self.candidates[index]

    def __iter__(self) -> Iterator[CandidateT]:
        return iter(self.candidates)


def scipy_slsqp_status(success: bool, code: int) -> tuple[ConvergenceState, FailureCause]:
    """把 SciPy SLSQP 的原生结束码翻译为领域状态。"""
    if success:
        return ConvergenceState.CONVERGED, FailureCause.NONE
    if code == 9:
        return ConvergenceState.MAX_ITERATIONS, FailureCause.MAX_ITERATIONS_REACHED
    if code in (5, 6, 7):
        return ConvergenceState.FAILED, FailureCause.SINGULAR_JACOBIAN
    if code in (3, 4, 8):
        return ConvergenceState.INFEASIBLE, FailureCause.CONSTRAINT_VIOLATION
    return ConvergenceState.FAILED, FailureCause.BACKEND_FAILURE


@dataclass(frozen=True)
class DifferentialCorrectionResult:
    """微分修正结果，软失败时可保留近似轨道。"""

    status: ConvergenceState
    cause: FailureCause
    message: str
    orbit: Orbit | None
    iterations: int
    residual: float | None
    residual_history: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@dataclass(frozen=True)
class ContinuationResult:
    """延拓结果，失败时仍可携带已生成的部分轨道族。"""

    status: ConvergenceState
    cause: FailureCause
    message: str
    family: OrbitFamily
    steps: int
    step_size: float

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@dataclass
class TransferCandidateResult:
    """转移搜索单格的类型化候选评估。

    承载搜索阶段对单个 ``(departure, alpha)`` 候选的全部几何与可行性信息。
    替代历史裸字典（ADR 0024）；``get``/``__getitem__`` 提供对字典风格
    消费方的兼容读取。
    """

    status: ConvergenceState
    cause: FailureCause
    message: str
    departure_state: np.ndarray | None = None
    departure_time: float = 0.0
    alpha: float = 0.0
    transfer_trajectory: Any | None = None
    transfer_times: Any | None = None
    transfer_time: float | None = None
    min_distance: float = float("inf")
    min_distance_idx: int | None = None
    min_distance_orbit_idx: int | None = None
    dv_departure: float = 0.0
    dv_insertion: float | None = None
    intersection_found: bool = False
    intersection_point: Any | None = None
    intersection_idx: int | None = None
    first_intersection_idx: int | None = None
    first_intersection_time: float | None = None
    first_min_distance_idx: int | None = None
    first_min_distance_time: float | None = None
    local_minimum_found: bool = False
    local_minimum_distance: float = float("inf")
    local_minimum_idx: int | None = None
    collision_found: bool = False
    collision_body: str | None = None
    collision_idx: int | None = None
    departure_orbit_name: str | None = None
    arrival_orbit_name: str | None = None
    departure_time_index: int | None = None

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)

    def get(self, key: str, default: Any = None) -> Any:
        """字典风格读取兼容。"""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """字典风格下标读取兼容。"""
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """字典风格成员检查兼容。"""
        return isinstance(key, str) and hasattr(self, key)


@dataclass(frozen=True)
class StageRecord:
    """任务可选阶段的适用性与执行记录。"""

    name: str
    applicable: bool
    executed: bool
    result_status: ConvergenceState | None
    message: str = ""

    def __post_init__(self) -> None:
        if self.result_status is ConvergenceState.ITERATING:
            raise ValueError("阶段记录不能以 ITERATING 结束")
        if not self.applicable and self.executed:
            raise ValueError("不适用阶段不能标记为已执行")
        if self.executed and self.result_status is None:
            raise ValueError("已执行阶段必须声明结果状态")
        if not self.executed and self.result_status is not None:
            raise ValueError("未执行阶段不能声明结果状态")


@dataclass(frozen=True)
class EphemerisCorrectionResult:
    """星历修正统一结果（Rust 多重打靶结果的领域重包）。

    Attributes:
        status: 收敛状态
        cause: 失败原因（收敛时为 NONE）
        message: 状态描述
        iterations: 迭代次数
        max_residual: 最终最大残差
        residual_history: 每次迭代最大残差的历史记录
        t_patch: 修正后的时间节点数组
        state_patch: 修正后的状态量数组
        velocity_residual: 速度残差（Rust 打靶 vel_weight 路径有值）
        velocity_residual_history: 速度残差历史（同上）
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
