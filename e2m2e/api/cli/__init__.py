"""命令行：人类入口。

CLI 子命令 = Facade 方法（mcp_exposed=True 的），参数从同一份 Pydantic 模型生成
（ADR 0014）。CLI 与 MCP 完全对称。

实现状态：完整——cli/main.py 已按工具清单（tool_inventory 的 implemented 条目）生成全部子命令。
"""

from __future__ import annotations

__all__: list[str] = []
