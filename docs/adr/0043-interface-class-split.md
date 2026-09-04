# ADR 0043: 接口类分家——Facade 只留任务级方法，轨道库与 spatiography 各自成类

**状态**：已采纳（已实现——类已分家、清单扫暴露类全集；`catalog_terminology`
随 ADR 0044 落地，`catalog_promote` 的移除随 ADR 0045 落地）
**日期**：2026-09-01
**相关 issue**：#610
**相关**：ADR 0011（五层架构）、ADR 0012（依赖方向）、ADR 0014（接口层——
决策 2 与决策 5 在此修订）、ADR 0031（决策 7 在此修订）、ADR 0035（sidecar
协议）、ADR 0041、ADR 0042（决策 5 的工具数措辞在此修订）、ADR 0044
（术语清单）、ADR 0045（记录粒度）。

## 背景

`Facade` 已经长到 31 个方法：6 个私有辅助、`config`、24 个
`@mcp_exposed` 方法（18 个已实现、6 个占位）。一个类混了三件不相干的事——
任务设计、轨道库数据管理、空间分区分析——共用一个名字。想设计一条轨道
的读者，必须翻过轨道库和分区方法才能找到那五个要紧的。

两条已记录的决策造就了这个形状。ADR 0014 决策 2 把 MCP 工具定义为
Facade 方法的全集，于是每个新能力要成为可调用，都得落在 `Facade` 上。
ADR 0031 决策 7 出于同样的理由把轨道库的查询与扫描方法放进了 Facade。
两条决策都没有预料到：这个类本身是要被人读的，如今它成了接口层最大的
理解成本。

相关症状：工具数在 ADR 0042 决策 5 和 issue 文本里被引作 22，而
`tool_specs(Facade())` 实测 18。靠数字守数字必然漂移；只有规则活得下来。

## 决策

### 1. Facade 只留任务级方法

`Facade` 恰好保留五个任务级能力：`design_orbit`、`control_orbit`、
`transfer_design`、`orbit_propagation`、`spacetime_transform`。私有辅助
与 `config` 留下。

### 2. 轨道库类

新的接口层类承接轨道库数据管理与族生成：`catalog_query`、
`catalog_get`、`catalog_delete`、`catalog_tag`、`catalog_export`、
`catalog_sweep`、`catalog_promote`（移除由 ADR 0045 决策 5 排期）、
`orbit_family_generation`，以及 ADR 0044 引入的术语清单方法。族生成放
这里，因为 `catalog_sweep` 就是它的批量编排——两者调用同一个
`run_family_sweep` 内核，且都产出轨道库记录。

### 3. spatiography 类

新的接口层类承接 `spatiography_scales`、`spatiography_classify`、
`spatiography_boundaries`、`spatiography_resonance_atlas`、
`spatiography_dynamical_map`。

### 4. 占位方法随领域归位

六个二档占位声明从 Facade 移除；它们落地时直接进领域类，不回 Facade。
`orbit_stability`、`manifold_analysis`、`relative_motion` 是分析能力；
`transfer_search`、`low_thrust_design`、`low_energy_transfer` 是转移
能力。各自的最终归属在实现时决定，不在本条决定。

### 5. 工具清单扫暴露类全集；清单仍单一来源

`tool_inventory()` 以暴露类实例全集为扫描根，返回一份扁平的
`ToolInfo` 列表。MCP 注册、CLI 子命令派生、sidecar preflight 照旧只消费
这一份清单——ADR 0014 决策 2 的机制保留，只是扫描根变宽。工具名、
schema、行为不变，MCP 与 CLI 调用方看不到任何差别。

### 6. 工具面准入判据取代工具数

ADR 0042 决策 5 的"工具数守 22"作为判据作废。新注册工具的准入条件：
要么它是任务级能力，要么它的内容被既有响应的字段引用、且没有既有工具
能供给（术语清单一例，ADR 0044）。工具数靠跑清单报告，不从文档里引用。

## 理由

1. **类分家而非缩面。** 曾考虑把 MCP 面缩到五个任务工具，否决：Agent
   会保留 ADR 0031 决策 8 的自动入库行为，却失去一切读回入库产物的
   途径；GUI 也会失去它的轨道库调用链。分家要治的理解成本落在"读类的
   人"身上，分家即可消除它，不必裁剪能力。
2. **族生成与轨道库同住，不与任务同住。** 它一次产出多条记录，且与
   `catalog_sweep` 共用内核；放在 `design_orbit` 旁边会把一个机制拆到
   两个类里。
3. **一份清单、多个扫描根。** 任何按类登记的注册表都会造出第二份清单，
   而 ADR 0014 决策 2 存在的意义就是防止第二份清单。
4. **规则，不是数字。** 22 对 18 的漂移证明散文里的数字会腐坏；准入
   判据在评审时可以核对。

## 后果

### 新增

- 两个接口层类（轨道库、spatiography）与放宽的 `tool_inventory()`
  扫描根。

### 变更

- ADR 0014 决策 2：MCP 工具 = 各暴露类 `mcp_exposed` 方法的并集。
  决策 5（CLI 对称）跟随同一并集。
- ADR 0031 决策 7：轨道库方法住轨道库类。
- ADR 0042 决策 5：工具数条款由本 ADR 决策 6 取代。
- 为轨道库或 spatiography 持有 `Facade` 的进程内调用方改用新类。
  transfer-orbit-design 的 `facade_bridge` 是已知唯一此类调用方。

### 不变

- 工具名、请求模型、信封、二进制帧契约、sidecar 协议、CLI 子命令名，
  以及写作当时的 18 工具面。
- 五层依赖方向；三个类都留在 `api/`。

### 代价

- 对进程内调用方是一次破坏性变更，他们得不到任何行为收益——收益属于
  接口层的读者。

## 实现注（2026-09-01，#610）

决策 5 的字面是清单"取暴露类实例全集"；实现取组合根再派生全集
（`Facade.exposed_apis`），理由有二：既有的构造与分发缝（CLI、sidecar、
MCP worker、测试）都传单个根对象；execution/sidecar 的测试桩注入的是
不带 `exposed_apis` 的单对象——解析器对无该属性的对象按自身扫描
（`e2m2e.api.facade.resolve_tool_method` 是属主解析的唯一入口）。扫描根
仍是暴露类实例全集，只是获取方式不同。

## 修订（2026-09-04，#620）

决策 1 的五方法清单扩为六：新增无参查询出口 `valid_ranges`（请求侧
条件值域全量导出，ADR 0014 决策 8 请求侧）。它无任务副作用、不属任何
领域类——横跨 design_orbit 与族生成的请求面元数据，挂组合根；其余
决策不变。
