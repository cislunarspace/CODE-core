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

from ..facade import resolve_tool_method, tool_inventory

if TYPE_CHECKING:
    pass

__all__ = ["ToolSpec", "tool_spec", "tool_specs"]


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    """一个可注册的 MCP 工具规格。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    method: Callable[..., Any]


def tool_spec(facade: Any, name: str) -> ToolSpec | None:
    """单个工具的规格：按需构造 schema，不重建全量清单（#601）。

    属主解析跨暴露类（ADR 0043 决策 5，单一入口
    :func:`~e2m2e.api.facade.resolve_tool_method`）；无 ``exposed_apis`` 的
    单一对象（测试桩）按自身解析。未知工具或 placeholder 返回 None——
    placeholder 不注册，调用未知工具同义。
    """
    attr = resolve_tool_method(facade, name)
    if attr is None or attr.tool_status != "implemented":
        return None
    return _build_spec(attr, name)


def tool_specs(facade: Any) -> list[ToolSpec]:
    """由暴露类纯派生工具规格（placeholder 不注册）。"""
    return [
        _build_spec(_resolve_method(facade, info.name), info.name)
        for info in tool_inventory(facade)
        if info.status == "implemented"
    ]


def _resolve_method(facade: Any, name: str) -> Any:
    """清单条目 → 属主实例上的绑定方法；无属主是清单与实现不同源的信号。"""
    attr = resolve_tool_method(facade, name)
    if attr is None:  # pragma: no cover - 清单与方法同源，防御分支
        raise LookupError(f"tool_inventory 条目无属主：{name}")
    return attr


def _build_spec(method: Any, name: str) -> ToolSpec:
    info_request_model = getattr(method, "request_model", None)
    if info_request_model is not None:
        schema = info_request_model.model_json_schema()
    else:
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
    return ToolSpec(
        name=name,
        description=(method.__doc__ or name).strip().splitlines()[0],
        input_schema=schema,
        method=method,
    )
