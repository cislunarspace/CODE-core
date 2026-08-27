# ADR 0014: Interface layer — Facade / MCP / CLI / 接口层 Facade/MCP/CLI

[English](#adr-0014-interface-layer--facade--mcp--cli) | [简体中文](#中文)

## English

**Status**: Adopted (Facade and MCP landed; full CLI subcommands still a
placeholder)
**Date**: 2026-07-31
**Related**: ADR 0011 (five-layer architecture), README vision
(LLM+Agent-callable)

### Context

e2m2e's current external shape is loose APIs (users assemble
`CR3BP_System → Dynamics → DifferentialCorrection`) with no unified entry, no
MCP, no CLI. README's vision requires LLMs to invoke Earth-Moon orbit
algorithms as they would Lambert or C-W tools. The interface layer (`api/`)
is where that vision gets delivered.

### Decision

1. **The Facade is the sole entry point**, its methods mapping to task-level
   capabilities (coarse granularity). The algorithm layer keeps fine-grained
   APIs for experts. Two granularity tiers.
2. **Pure derivation + metadata markers**: MCP tools = the complete set of
   Facade methods; Facade methods carry `mcp_exposed: bool` metadata (tier-1/
   tier-2 True; tier-3/auxiliary False). Registration scans Facade methods;
   the list has one source of truth.
3. **Pydantic models all hand-written**: input/output/error models carefully
   specify parameter units, defaults, value domains. They stay at the `api/`
   boundary, never entering the algorithm layer.
4. **Facade returns dedicated Pydantic models**; the MCP transport wraps a
   uniform envelope ({status, data, error, meta}). Errors are translated in
   `api/`: exceptions → structured error codes (OrbitError with code/message/
   details).
5. **CLI subcommands = Facade methods** (those with mcp_exposed=True),
   parameters generated from the same Pydantic models. CLI and MCP are fully
   symmetric.
6. **MCP deployment = in-process library as the main body + thin CLI wrapper
   `mcp-serve`**: `create_server(facade)` function + `e2m2e mcp-serve`
   subcommand. One Facade instance = one server.
7. **config.py constructor injection**: `Facade(config=Config(...))`,
   covering only runtime environment (kernel paths/precision thresholds/
   logging); physical constants belong to data/templates/. SPICEManager's
   global handles and r2s2's process singleton are known limitations managed
   explicitly via Config.
8. **Conditional value domains are public and single-sourced**: input-model
   value domains depending on other fields must be exposed through
   machine-readable public interfaces; validators and those interfaces share
   one rule definition. GUIs, CLIs, and MCP must not parse error text, read
   validator source, or maintain local copies of ranges.

### MCP tool list

- Tier 1 task-level (stable skeleton, will grow): design_orbit / control_orbit
  / transfer_design / orbit_propagation / spacetime_transform.
- Tier 2 subtask-level (will grow): orbit_family_generation / orbit_stability
  / transfer_search / low_thrust_design / manifold_analysis /
  low_energy_transfer / relative_motion.
- Tier 3 auxiliary (not registered): porkchop / normal_form / safety /
  visualize / format I/O.

### Rationale

1. **Pure derivation**: Facade methods are the single source; adding a
   capability = adding a Facade method + hand-written model = MCP tool + CLI
   subcommand both appear automatically; the list never drifts.
2. **Hand-written models**: tier-1 is what Agents use most; schemas need care
   (units/defaults/domains written plainly in Pydantic); where conditional
   domains exceed static schema expressiveness, model public interfaces
   supplement them — for long-term maintenance quality.
3. **CLI↔MCP symmetry**: the same Facade method serves MCP for Agents and CLI
   for humans, validated by the same model set.

### Consequences

- The `api/` layer provides Facade/config/models/mcp/cli.
- transfer-orbit-design keeps an independent repo hosting only the GUI
  (deprecating tod/generates algorithm script layers, superseded by e2m2e
  CLI); GUI parameter forms are generated from e2m2e Pydantic models and their
  conditional-domain public interfaces.

## 中文

**状态**：已采纳（Facade 与 MCP 已落地；CLI 完整子命令仍占位）
**日期**：2026-07-31
**关联**：ADR 0011（五层架构）、README 愿景（LLM+Agent 可调用）

### 背景

e2m2e 现有对外形态是散装 API（用户拼 `CR3BP_System → Dynamics → DifferentialCorrection`），无统一入口、无 MCP、无 CLI。README 愿景要求大模型可以像调用 Lambert、C-W 工具一样调用地月轨道算法。接口层（api/）是这一愿景的兑现点。

### 决策

1. **Facade 是唯一入口**，方法对应任务级能力（粗粒度）。算法层保留细粒度 API（专家用）。两层粒度。
2. **纯派生 + 元数据标记**：MCP 工具 = Facade 方法全集；Facade 方法带 `mcp_exposed: bool` 元数据（一档二档 True、三档/辅助 False）。注册逻辑统一扫 Facade 方法，清单单一来源。
3. **Pydantic 模型全部手写**：输入/输出/错误模型精雕参数单位、默认值、取值域。只在 api/ 边界，不进算法层。
4. **Facade 返回专属 Pydantic 模型**；MCP 传输层包统一信封（{status, data, error, meta}）。错误在 api/ 翻译：异常 → 结构化错误码（OrbitError 含 code/message/details）。
5. **CLI 子命令 = Facade 方法**（mcp_exposed=True 的），参数从同一份 Pydantic 模型生成。CLI 与 MCP 完全对称。
6. **MCP 部署形态 = 进程内库为主体 + CLI 薄包装 mcp-serve**：`create_server(facade)` 函数 + `e2m2e mcp-serve` 子命令。一个 Facade 实例 = 一个 server。
7. **config.py 构造注入**：`Facade(config=Config(...))`，只管运行环境（内核路径/精度阈值/日志）；物理常量归 data/templates/。SPICEManager 全局句柄、r2s2 进程单例作为已知限制用 Config 显式管理。
8. **条件取值域公开且同源**：输入模型中依赖其他字段的取值域，必须通过机器可读的公开接口提供；校验器与该接口共用一份规则定义。GUI、CLI、MCP 不得解析错误文本、阅读校验器源码或维护本地范围副本。

### MCP 工具清单

- 一档任务级（稳定骨架，会增）：design_orbit / control_orbit / transfer_design / orbit_propagation / spacetime_transform。
- 二档子任务级（会增）：orbit_family_generation / orbit_stability / transfer_search / low_thrust_design / manifold_analysis / low_energy_transfer / relative_motion。
- 三档辅助（不注册）：porkchop / normal_form / safety / visualize / 格式读写。

### 理由

1. **纯派生**：Facade 方法单一来源，加能力 = 加 Facade 方法 + 手写模型 = MCP 工具 + CLI 子命令自动都有，清单不漂移。
2. **手写模型**：一档是 Agent 最常用的，schema 要精心（单位/默认值/取值域在 Pydantic 里写清）；条件取值域不能由静态 schema 完整表达时，以模型公开接口补充，为后续维护质量。
3. **CLI 与 MCP 对称**：同一个 Facade 方法，MCP 给 Agent、CLI 给人类，参数校验同一套模型。

### 结果

- api/ 层提供 Facade/config/models/mcp/cli。
- transfer-orbit-design 保留独立仓库只留 GUI（废弃 tod/generates 算法脚本层，被 e2m2e CLI 覆盖），GUI 参数表单从 e2m2e Pydantic 模型及其条件取值域公开接口生成。
