"""接口层组件目录。"""

from .components import Component

API_COMPONENTS = [
    Component(
        name="OrbitDesignFacade",
        module_path="e2m2e.api.facade",
        layer="api",
        description="Task-level orbit design facade",
    ),
    Component(
        name="CLI",
        module_path="e2m2e.api.cli",
        layer="api",
        description="Command-line interface",
    ),
    Component(
        name="MCPServer",
        module_path="e2m2e.api.mcp",
        layer="api",
        description="MCP server interface",
    ),
]
