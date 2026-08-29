"""Unified result-status contract shared by every layer.

ADR 0024's status vocabulary—``ConvergenceState``, ``FailureCause``,
``ResultStatus`` and the ``CAUSE_STATUS`` mapping—is consumed by the data,
algorithm and numerics layers alike, so its single home is a package-root
shared-kernel leaf: importable by every layer, importing no layer itself
(ADR 0039). The historical paths ``e2m2e.data.templates`` and
``e2m2e.algorithm.results`` re-export these objects with unchanged identity.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

__all__ = ["CAUSE_STATUS", "ConvergenceState", "FailureCause", "ResultStatus"]


class ConvergenceState(enum.Enum):
    """算法最终状态。"""

    ITERATING = "iterating"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    STAGNATED = "stagnated"
    MAX_ITERATIONS = "max_iterations"
    INFEASIBLE = "infeasible"
    COLLISION = "collision"
    FAILED = "failed"


class FailureCause(enum.Enum):
    """算法最终结局的稳定原因码。"""

    NONE = "none"
    INTEGRATION_FAILED = "integration_failed"
    SINGULAR_JACOBIAN = "singular_jacobian"
    INVALID_PERIOD = "invalid_period"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    STAGNATION_DETECTED = "stagnation_detected"
    DIVERGENCE_DETECTED = "divergence_detected"
    NO_INTERSECTION = "no_intersection"
    CONSTRAINT_VIOLATION = "constraint_violation"
    BODY_COLLISION = "body_collision"
    LEVEL1_CORRECTION_FAILED = "level1_correction_failed"
    BACKEND_FAILURE = "backend_failure"
    INVALID_INPUT = "invalid_input"
    UNKNOWN = "unknown"


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
