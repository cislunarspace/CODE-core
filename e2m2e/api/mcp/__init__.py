"""MCP 服务子包（ADR 0014）。

- ``envelope``：统一信封 {status, data, error, meta} 与异常翻译。
- ``tools``：由 Facade 纯派生的工具规格（placeholder 不注册）。
- ``worker``：长任务 worker 子进程（JSON 行协议，不依赖 ``[mcp]`` extra）。
- ``server``：``create_server(facade)``（依赖 ``[mcp]`` extra）。

``server`` 惰性导出：sidecar（ADR 0035）复用 envelope/tools 但不依赖
``[mcp]`` extra，本包在缺 ``mcp`` 库时仍可导入。
"""

from __future__ import annotations

from typing import Any

from .envelope import error_envelope, invoke_tool, ok_envelope
from .tools import ToolSpec, tool_specs

__all__ = [
    "LONG_RUNNING_TOOLS",
    "ToolSpec",
    "create_server",
    "error_envelope",
    "handle_call_tool",
    "handle_list_tools",
    "invoke_tool",
    "ok_envelope",
    "run_tool_in_worker",
    "tool_specs",
]

_LAZY = (
    "LONG_RUNNING_TOOLS",
    "create_server",
    "handle_call_tool",
    "handle_list_tools",
    "run_tool_in_worker",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
