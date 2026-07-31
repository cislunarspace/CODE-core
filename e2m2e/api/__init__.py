"""接口层：Facade 门面、配置、Pydantic 模型、MCP、CLI。

第 4 层，依赖方向：algorithm/ + data/（ADR 0012）。Pydantic 只在 api/ 边界，
算法层用 numpy/dataclass。

- ``facade.py``：Facade 门面，唯一公开顶级入口，粗粒度任务方法。
- ``config.py``：配置（只管运行环境：内核路径/精度阈值/日志）。
- ``models.py``：公开数据模型（Pydantic，全手写）。
- ``mcp/``：MCP 服务（create_server(facade) 进程内 + CLI mcp-serve 薄包装）。
- ``cli/``：命令行（子命令 = Facade 方法）。

实现状态：骨架。Facade 方法待接入 algorithm/ 各编排器后逐能力实现。
"""

__all__: list[str] = []
