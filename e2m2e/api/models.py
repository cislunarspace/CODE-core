"""公开数据模型（Pydantic，全手写）。

输入/输出/错误模型精雕参数单位、默认值、取值域（ADR 0014）。只在 api/ 边界。
模型字段清单在实现阶段细化（每个 Facade 方法一个 Request/Response + OrbitError）。

实现状态：骨架。字段清单待定稿。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["OrbitError"]


@dataclass
class OrbitError(Exception):
    """结构化错误（api/ 边界翻译，ADR 0014）。

    Attributes:
        code: 错误码（如 "NOT_IMPLEMENTED"/"NOT_CONVERGED"）。
        message: 可读错误信息。
        details: 附加细节。
    """

    code: str = "ERROR"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
