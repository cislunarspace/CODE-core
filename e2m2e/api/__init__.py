"""接口层：任务级入口、领域接口类、配置、Pydantic 模型、MCP、CLI。

第 4 层，依赖方向：algorithm/ + data/（ADR 0012）。Pydantic 只在 api/ 边界，
算法层用 numpy/dataclass。

- ``facade.py``：Facade 任务级入口与组合根（五个任务方法；经
  ``.catalog`` / ``.spatiography`` 交出领域类，ADR 0043）。
- ``catalog.py``：Catalog 轨道库类（数据管理 + 族生成，ADR 0043 决策 2）。
- ``spatiography.py``：Spatiography 分区分析类（ADR 0043 决策 3）。
- ``config.py``：配置（只管运行环境：内核路径/精度阈值/日志）。
- ``models.py``：公开数据模型（Pydantic，全手写）。
- ``mcp/``：MCP 服务（create_server(facade) 进程内 + CLI mcp-serve 薄包装）。
- ``cli/``：命令行（子命令 = 工具清单 implemented 条目）。

MCP 工具 = 各暴露类 ``mcp_exposed`` 方法并集，单一清单
（``tool_inventory``，ADR 0014 决策 2 经 ADR 0043 拓宽扫描根）。

仓库全貌与一条任务链的走读见 README 的仓库怎么读一节。
"""

from .catalog import Catalog
from .facade import Facade, ToolInfo, mcp_tools, tool_inventory
from .models import (
    ControlOrbitRequest,
    ControlOrbitResponse,
    DesignOrbitRequest,
    DesignOrbitResponse,
    NumericRange,
    OrbitError,
)
from .spatiography import Spatiography

__all__ = [
    "Facade",
    "Catalog",
    "Spatiography",
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
