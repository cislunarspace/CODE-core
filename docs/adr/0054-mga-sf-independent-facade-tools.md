# ADR 0054: 行星际 MGA 与 Sims-Flanagan 预设计以独立 Facade 任务级工具暴露

**状态**：已采纳（决策记录；算法与接口实现随 #725/#726 执行链交付）
**日期**：2026-09-27
**相关 issue**：#721（本决策 issue）、#725（Sims-Flanagan conic 档）、#726（segment 框架）、#723（MGA 连接器）
**相关**：ADR 0040（transfer_design 收敛契约——语义不兼容，不扩）、ADR 0043
（接口类分家；决策 4 的 `low_thrust_design` 占位归属在此落定）、ADR 0044
（术语清单冻结）、ADR 0050（SF 双档保真度）、ADR 0053（通用段框架后置归纳）

## 背景

Issue #721 提出行星际多借力（MGA）与 Sims-Flanagan（SF）预设计如何进入
调用方接口，三个选项：

- **A. 扩 `transfer_type`**：在既有 `transfer_design` 的转移类型清单里加
  MGA/SF 档，复用其请求/响应与目录链路。
- **B. 独立 Facade 任务级工具**：MGA 链/发射窗口搜索与 SF 预设计各自成为
  独立任务级工具。
- **C. 暂不暴露**：算法层先行，接口后议。

代码与决策事实（#721 会话源码核实，2026-09-27）：

- `transfer_type` 是术语清单：`TRANSFER_TYPES = ("HMN", "LGA", "WSB",
  "low_thrust", "PCN")`（`e2m2e/data/catalog/terminology.py`），随包版本
  冻结（ADR 0044）；`TransferDesignRequest.transfer_type` 为自由 str，
  字段描述列上述五值。
- ADR 0040 的 `transfer_design` 收敛契约是地月语义：`trajectory_times`
  自 TLI 起算（t=0 为出发脉冲）、`state_frame` 词汇以地月会合系为主、
  `maneuver_events` 的 kind 枚举为 departure/perilune/arrival。日心多
  leg 任务没有 TLI 时刻，语义不兼容。
- ADR 0043 决策 4 把 `low_thrust_design` 等二档占位从 Facade 移除，最终
  归属"在实现时决定"。
- ADR 0050 已决 SF 工具链双档保真度（conic 快筛档/星历 n 体档），conic
  档随 #725 交付；ADR 0053 把通用段框架后置到 #726 一次性归纳。
- 长任务路由是单一清单：`LONG_RUNNING_TOOLS = frozenset({"transfer_design",
  "orbit_family_generation"})`（`e2m2e/api/execution.py`），清单内工具经
  worker 子进程执行、可取消，MCP/CLI/sidecar 三个传输层共用。
- 仓内尚无任何 MGA/Sims-Flanagan 代码，无既有接口形状需要兼容。

## 决策

1. **选 B：独立 Facade 任务级工具，不扩 `transfer_type` 清单，不改
   ADR 0040 契约。** MGA/SF 预设计是任务级能力（ADR 0043 决策 6 准入
   判据第一类），以新方法带 `@mcp_exposed` 进 `tool_inventory()` 单一
   清单，MCP/CLI/sidecar 照常派生；接口类归属遵循 ADR 0043 决策 4 在
   实现时落位，不回 Facade 占位。
2. **定名两个工具**：`mission_architecture_search`（MGA 链/发射窗口
   搜索）与 `low_thrust_preliminary`（Sims-Flanagan 预设计，下辖
   ADR 0050 的 conic 档与后续星历档）。命名遵循仓库请求/响应对约定，
   为将来的 `MissionArchitectureSearchRequest`/`Response`、
   `LowThrustPreliminaryRequest`/`Response` 留名；`low_thrust_preliminary`
   与既有 `transfer_type="low_thrust"`（地月低推力转移）字面可区分。
3. **长任务路由**：两工具实施时加入 `LONG_RUNNING_TOOLS`，经 worker
   子进程执行，MCP/CLI/sidecar 三个传输层自动获得隔离与取消；不新增
   第二份路由清单。
4. **新工具收敛结果首版不入 catalog**：记录侧的 `transfer_type`、
   `state_frame`、`maneuver_events` 词汇均为地月语义，日心多 leg 结果
   无对应键可填；入库问题（记录形态、索引键、可查询面）随 #726 段框架
   再定。

## 理由

1. **语义不兼容否决 A**：ADR 0040 契约的时间基准（TLI 秒）、数据系
   词汇（`synodic_barycentric_km` 等）、机动事件枚举
   （departure/perilune/arrival）都预设地月场景；日心多 leg 响应若硬套
   该契约，要么伪造 TLI 时刻，要么逐字段重开语义——扩值省下的接口设计
   会以契约腐蚀的形式偿还。
2. **冻结清单否决 A 的词表部分**：`transfer_type` 属 ADR 0044 术语
   清单，扩值即破坏"只随包发布变更"的版本冻结保证，且所有按值分支的
   调用方行为面同时受影响。
3. **粒度一致**：ADR 0043 把接口整理为任务级能力；MGA 链搜索与 SF
   预设计各是一个完整任务（输入任务参数、输出候选方案），符合该粒度，
   也直接通过决策 6 的准入判据。
4. **长任务归长任务**：SF 广泛搜索与 MGA 网格扫描的计算特征与既有两个
   长任务工具相同（分钟级、需取消）；进 `LONG_RUNNING_TOOLS` 是消费
   既有机制而非新造机制。
5. **C 否决**：不暴露则 #725/#726 的交付无法被 MCP/CLI 调用方触达，
   ADR 0043 的占位继续悬置；先锁决策记录可以把命名与路由契约定死，
   实现链零歧义。

## 后果

- `transfer_type` 术语清单保持五值不变；本 ADR 吸收 ADR 0043 决策 4 中
  `low_thrust_design` 占位的归属（落定为 `low_thrust_preliminary`）。
- 本决策无代码行为变化；`CHANGELOG.md` 仅记录决策本身的调用方知会，
  行为变化条目随 #725/#726 实现时再记。
- #725 实施时以 `low_thrust_preliminary` 暴露 conic 档并加入
  `LONG_RUNNING_TOOLS`；`mission_architecture_search` 随 MGA 实现链
  （#723/#726）同法暴露。

## 备选方案

- **A 扩 `transfer_type`（`transfer_design` 下加 MGA/SF 档）**：复用面
  最大，但契约语义不兼容（理由 1）且冻结清单不容扩值（理由 2）；否决。
- **C 暂不暴露（仅算法层内部）**：调用方无入口、占位悬置；否决。
- **顺手扩 ADR 0040 契约以兼容日心语义**：动收敛契约波及全部既有
  `transfer_design` 调用方，换来与本决策无关的破坏；否决，日心语义
  另立契约。
