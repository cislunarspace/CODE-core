"""共享 API 模型基础设施：统一错误模型、失败详情与响应基类。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from e2m2e.status import ConvergenceState, FailureCause, ResultStatus

__all__ = [
    "OrbitError",
    "propagation_failure_details",
    "ResultResponse",
]


class OrbitError(Exception):
    """结构化错误（api/ 边界翻译，ADR 0014）。

    Attributes:
        code: 错误码（如 "NOT_IMPLEMENTED"/"NOT_CONVERGED"/"INVALID_PARAMS"）。
        message: 可读错误信息。
        details: 附加细节。
    """

    def __init__(
        self,
        code: str = "ERROR",
        message: str = "",
        details: dict[str, Any] | None = None,
        status: ConvergenceState = ConvergenceState.FAILED,
        cause: FailureCause = FailureCause.UNKNOWN,
    ) -> None:
        super().__init__(message)
        ResultStatus(status, cause, message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.status = status
        self.cause = cause

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def propagation_failure_details(
    message: str,
    status: ConvergenceState | None = None,
    cause: FailureCause | None = None,
) -> dict[str, Any]:
    """传播失败的 ``error.details`` 载荷（#677 验收：传输层翻译零丢失）。

    诊断文本即算法/Rust 侧给出的 ``cause:`` 段（含星历缓存窗口的 et 与区间等
    定位字段），原样携带、不解析、不改写（ADR 0014 决策 8：传输层不以错误文本做
    翻译决策）。
    状态三元组缺省时按 ``FAILED`` / ``UNKNOWN`` 兜底。

    Args:
        message: 诊断文本（含 ``cause:`` 段）。
        status: 收敛状态；None 兜底 ``ConvergenceState.FAILED``。
        cause: 失败原因码；None 兜底 ``FailureCause.UNKNOWN``。

    Returns:
        ``{"status": <ConvergenceState 值>, "cause": <FailureCause 值>,
        "diagnostic": <message>}``。
    """
    return {
        "status": (status or ConvergenceState.FAILED).value,
        "cause": (cause or FailureCause.UNKNOWN).value,
        "diagnostic": message,
    }


class _ApiModel(BaseModel):
    """api 模型公共配置：允许任意内部字段，输出按声明序列化。"""

    model_config = ConfigDict(extra="forbid")


class ResultResponse(_ApiModel):
    """Facade 成功处理后的任务最终状态三元组。"""

    status: ConvergenceState
    cause: FailureCause
    message: str

    @model_validator(mode="after")
    def _validate_result_status(self) -> ResultResponse:
        ResultStatus(self.status, self.cause, self.message)
        return self
