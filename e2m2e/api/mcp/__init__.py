"""MCP 服务子包（ADR 0014）。

- ``envelope``：统一信封 {status, data, error, meta} 与异常翻译。
- ``tools``：由 Facade 纯派生的工具规格（placeholder 不注册）。
- ``server``：``create_server(facade)``（依赖 ``[mcp]`` extra）。
"""

from __future__ import annotations

from .envelope import error_envelope, invoke_tool, ok_envelope
from .server import create_server, handle_call_tool, handle_list_tools
from .tools import ToolSpec, tool_specs

__all__ = [
    "ToolSpec",
    "create_server",
    "error_envelope",
    "handle_call_tool",
    "handle_list_tools",
    "invoke_tool",
    "ok_envelope",
    "tool_specs",
]
