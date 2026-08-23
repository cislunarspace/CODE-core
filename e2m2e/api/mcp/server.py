"""MCP 服务：LLM 工具入口。

进程内库为主体 + CLI 薄包装 mcp-serve（ADR 0014）：``create_server(facade)``
函数（进程内、可测试）+ ``e2m2e mcp-serve`` 子命令。一个 Facade 实例 = 一个
server。MCP 工具 = facade 上 mcp_exposed=True 的方法（纯派生，见 tools.py），
传输层包统一信封（见 envelope.py）。

依赖 ``[mcp]`` extra：本模块在缺 ``mcp`` 库时导入即失败，调用方（CLI）负责
给出安装提示。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anyio.to_thread
from mcp.server import Server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

from . import envelope, tools

if TYPE_CHECKING:
    from ..facade import Facade

__all__ = ["create_server", "handle_list_tools", "handle_call_tool"]


def handle_list_tools(facade: Facade) -> list[Tool]:
    """列出工具（纯函数，便于测试）。"""
    return [
        Tool(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
        )
        for spec in tools.tool_specs(facade)
    ]


def _to_result(env: envelope.Envelope) -> CallToolResult:
    """信封 → MCP CallToolResult（JSON 文本 + isError 标志）。"""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(env, ensure_ascii=False))],
        is_error=env["status"] == "error",
    )


def handle_call_tool(facade: Facade, name: str, arguments: dict[str, Any]) -> CallToolResult:
    """调用工具并包信封（纯函数，便于测试）。"""
    spec = next((s for s in tools.tool_specs(facade) if s.name == name), None)
    if spec is None:
        env = envelope.error_envelope(
            "TOOL_NOT_FOUND", f"未知工具 {name!r}（placeholder 或未暴露的方法不注册）"
        )
    else:
        env = envelope.invoke_tool(spec.method, arguments)
    return _to_result(env)


def create_server(facade: Facade) -> Server:
    """创建 MCP 服务器（绑定传入的 Facade）。

    一个 Facade 实例 = 一个 server；工具清单在每次 tools/list 时由 Facade
    纯派生，与 ``tool_inventory()`` 单一同源。注册走
    ``add_request_handler`` （mcp 1.x/2.x 兼容：2.0 移除了装饰器 API）。

    Args:
        facade: Facade 实例。

    Returns:
        ``mcp.server.Server`` 对象（配合 ``mcp.server.stdio.stdio_server`` 运行）。
    """
    server: Server = Server("e2m2e")

    async def _list_tools(context: Any, params: Any) -> ListToolsResult:
        # 纯派生不走 Facade 方法本体，直接在事件循环里做即可。
        return ListToolsResult(tools=handle_list_tools(facade))

    async def _call_tool(context: Any, params: Any) -> CallToolResult:
        # Facade 方法是同步长计算，放线程池避免阻塞事件循环。
        arguments = dict(params.arguments or {})
        env = await anyio.to_thread.run_sync(
            lambda: handle_call_tool(facade, params.name, arguments)
        )
        return env

    server.add_request_handler("tools/list", PaginatedRequestParams, _list_tools)
    server.add_request_handler("tools/call", CallToolRequestParams, _call_tool)
    return server
