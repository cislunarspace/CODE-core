"""接口层组件目录。"""

from .components import Component

API_COMPONENTS = [
    Component(
        name="OrbitDesignFacade",
        module_path="e2m2e.api.facade",
        layer="api",
        description="任务级轨道设计门面",
    ),
    Component(
        name="CLI",
        module_path="e2m2e.api.cli",
        layer="api",
        description="命令行接口",
    ),
    Component(
        name="MCPServer",
        module_path="e2m2e.api.mcp",
        layer="api",
        description="MCP 服务接口",
    ),
]
