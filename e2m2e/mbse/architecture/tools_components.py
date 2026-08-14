"""工具层组件目录。"""

from .components import Component

TOOLS_COMPONENTS = [
    Component(
        name="LoggingTools",
        module_path="e2m2e.tools.logging",
        layer="tools",
        description="结构化日志辅助工具",
    ),
]
