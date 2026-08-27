"""接口层：Facade 门面、配置、Pydantic 模型、MCP、CLI。

第 4 层，依赖方向：algorithm/ + data/（ADR 0012）。Pydantic 只在 api/ 边界，
算法层用 numpy/dataclass。

- ``facade.py``：Facade 门面，唯一公开顶级入口，粗粒度任务方法。
- ``config.py``：配置（只管运行环境：内核路径/精度阈值/日志）。
- ``models.py``：公开数据模型（Pydantic，全手写）。
- ``mcp/``：MCP 服务（create_server(facade) 进程内 + CLI mcp-serve 薄包装）。
- ``cli/``：命令行（子命令 = Facade 方法）。

实现状态：一档任务（design/control）与二档子任务（family/stability/proximity）
已接入 algorithm/；transfer_design/orbit_propagation/spacetime_transform 及
MCP/CLI 依赖 [mcp] extra，保持占位。

仓库全貌与一条任务链的走读见 README 的仓库怎么读一节。
"""

from .facade import Facade, ToolInfo, mcp_tools, tool_inventory
from .models import (
    ControlOrbitRequest,
    ControlOrbitResponse,
    DesignOrbitRequest,
    DesignOrbitResponse,
    NumericRange,
    OrbitError,
)

__all__ = [
    "Facade",
    "ToolInfo",
    "mcp_tools",
    "tool_inventory",
    "OrbitError",
    "NumericRange",
    "DesignOrbitRequest",
    "DesignOrbitResponse",
    "ControlOrbitRequest",
    "ControlOrbitResponse",
]
