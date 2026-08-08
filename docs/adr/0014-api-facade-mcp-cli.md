# ADR 0014：接口层 Facade/MCP/CLI

**状态**：已接受（部分实施：Facade 已落地，MCP/CLI 占位）
**日期**：2026-07-31
**关联**：ADR 0011（五层架构）、README 愿景（LLM+Agent 可调用）

## 背景

e2m2e 现有对外形态是散装 API（用户拼 `CR3BP_System → Dynamics → DifferentialCorrection`），无统一入口、无 MCP、无 CLI。README 愿景要求"大模型可以像调用 Lambert、C-W 工具一样调用地月轨道算法"。接口层（api/）是这一愿景的兑现点。

## 决策

1. **Facade 是唯一入口**，方法对应"任务级能力"（粗粒度）。算法层保留细粒度 API（专家用）。两层粒度。
2. **纯派生 + 元数据标记**：MCP 工具 = Facade 方法全集；Facade 方法带 `mcp_exposed: bool` 元数据（一档二档 True、三档/辅助 False）。注册逻辑统一扫 Facade 方法，清单单一来源。一档也会增加。
3. **Pydantic 模型全部手写**：输入/输出/错误模型精雕参数单位、默认值、取值域。只在 api/ 边界，不进算法层。
4. **Facade 返回专属 Pydantic 模型**；MCP 传输层包统一信封（{status, data, error, meta}）。错误在 api/ 翻译：异常 → 结构化错误码（OrbitError 含 code/message/details）。
5. **CLI 子命令 = Facade 方法**（mcp_exposed=True 的），参数从同一份 Pydantic 模型生成。CLI 与 MCP 完全对称。
6. **MCP 部署形态 = 进程内库为主体 + CLI 薄包装 mcp-serve**：`create_server(facade)` 函数 + `e2m2e mcp-serve` 子命令。一个 Facade 实例 = 一个 server。
7. **config.py 构造注入**：`Facade(config=Config(...))`，只管运行环境（内核路径/精度阈值/日志）；物理常量归 data/templates/。SPICEManager 全局句柄、r2s2 进程单例作为已知限制用 Config 显式管理。

### MCP 工具清单

- 一档任务级（稳定骨架，会增）：orbit_design / orbit_control / transfer_design / orbit_propagation / spacetime_transform。
- 二档子任务级（会增）：orbit_family_generation / orbit_stability / transfer_search / low_thrust_design / manifold_analysis / low_energy_transfer / relative_motion。
- 三档辅助（不注册）：porkchop / normal_form / safety / visualize / 格式读写。

## 理由

1. **纯派生**：Facade 方法单一来源，加能力 = 加 Facade 方法 + 手写模型 = MCP 工具 + CLI 子命令自动都有，清单不漂移。
2. **手写模型**：一档是 Agent 最常用的，schema 要精心（单位/默认值/取值域在 Pydantic 里写清），为后续维护质量。
3. **CLI 与 MCP 对称**：同一个 Facade 方法，MCP 给 Agent、CLI 给人类，参数校验同一套模型。

## 结果

- api/ 层提供 Facade/config/models/mcp/cli。
- transfer-orbit-design 保留独立仓库只留 GUI（废弃 tod/generates 算法脚本层，被 e2m2e CLI 覆盖），GUI 参数表单从 e2m2e Pydantic 模型生成。
