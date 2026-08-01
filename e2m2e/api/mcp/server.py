"""MCP 服务：LLM 工具入口。

进程内库为主体 + CLI 薄包装 mcp-serve（ADR 0014）：``create_server(facade)``
函数（进程内、可测试）+ ``e2m2e mcp-serve`` 子命令。一个 Facade 实例 = 一个
server。MCP 工具 = facade 上 mcp_exposed=True 的方法（纯派生），传输层包统一信封
（{status, data, error, meta}）。

实现状态：骨架。协议层依赖 ``[mcp]`` extra。
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_server"]


def create_server(facade) -> Any:
    """创建 MCP 服务器（绑定传入的 Facade）。

    实现状态：待实现（依赖 [mcp] extra）。

    Args:
        facade: Facade 实例。

    Returns:
        MCP server 对象。
    """
    raise NotImplementedError("MCP create_server 待实现（依赖 [mcp] extra）")
