"""MCP 工具注册：由 Facade 方法自动派生。

纯派生 + 元数据标记（ADR 0014）：MCP 工具 = Facade 方法全集，凡 mcp_exposed=True
的方法都注册。清单单一来源，一档也会增加。

实现状态：骨架。待 create_server 落地后实现派生逻辑。
"""

from __future__ import annotations

__all__: list[str] = []
