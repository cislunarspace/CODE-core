"""MCP 工具注册：由 Facade 方法自动派生。

纯派生 + 元数据标记（ADR 0014）：MCP 工具 = Facade 方法全集，凡
``mcp_exposed=True`` 的方法都注册。清单单一来源是 ``tool_inventory()``，
本模块只消费它，不维护第二份清单。

placeholder 状态的工具**不注册**：Agent 不应调到空实现；待其落地为
implemented 后由清单单一来源自动出现在 server 上。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..facade import tool_inventory

if TYPE_CHECKING:
    from ..facade import Facade

__all__ = ["ToolSpec", "tool_specs"]


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    """一个可注册的 MCP 工具规格。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    method: Callable[..., Any]


def tool_specs(facade: Facade) -> list[ToolSpec]:
    """由 Facade 纯派生工具规格（placeholder 不注册）。"""
    specs: list[ToolSpec] = []
    for info in tool_inventory(facade):
        if info.status != "implemented":
            continue
        method = getattr(facade, info.name)
        if info.request_model is not None:
            schema = info.request_model.model_json_schema()
        else:
            schema = {"type": "object", "properties": {}, "additionalProperties": False}
        specs.append(
            ToolSpec(
                name=info.name,
                description=(method.__doc__ or info.name).strip().splitlines()[0],
                input_schema=schema,
                method=method,
            )
        )
    return specs
