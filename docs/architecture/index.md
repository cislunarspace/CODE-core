# 架构

e2m2e 是"LLM+Agent"任务规划系统中的**算法工具集基础设施**：大模型负责理解意图与编排，e2m2e 负责精确可靠的轨道计算。本文是架构章节的阅读地图。

## 五个架构设计

完整叙述见 [architecture](architecture.md)，摘要如下：

| 模块 | 职责 | 关键决策 |
|---|---|---|
| 时空系统与常量 | UTC/TDB/TAI/TT 时间尺度、J2000/ITRF93/GCRS 等参考系转换、多套物理常量基准集 | ADR 0010、ADR 0022 |
| Rust 计算 | 四个核心 crate（spice / propagation / forces / integrators），另有 HJB 求解器两 crate（levelset / hjb-dynamics，见 [hjb-subsystem](hjb-subsystem.md)） | ADR 0002、ADR 0016、ADR 0032 |
| Python 编排 | 构造问题 → 调 Rust 迭代器 → 解释结果；任务级 Facade | ADR 0014、ADR 0029 |
| CI | 三平台 wheel 矩阵（Linux x64/ARM、Windows）+ CSPICE 编译包分发 | ADR 0009 |
| 数据管理 | GitHub Release 星历数据、Git 跟踪族种子、随包 CR3BP 基线族数据集、本地 catalog | ADR 0031、ADR 0036 |

分工原则（ADR 0011 五层架构、ADR 0012 依赖方向）：**领域决策留 Python，热循环进 Rust**。Rust 不吃 SPICE 句柄，吃预采样注入的星历缓存表——cspice 内核是全局状态、不可并发，这一约束决定了接缝的位置。

## 章节阅读地图

- [architecture](architecture.md) — 总览：从设计一条 L2 NRHO 的完整链路看五个模块各自在哪个环节发挥作用。先读这篇。
- [system-dynamics-dataflow](system-dynamics-dataflow.md) — System 与 Dynamics 两棵类层次的深潜：构造、传播、结果缓存中数据逐段怎么走。
- [numerics-migration-status](numerics-migration-status.md) — algorithm 层各子模块的数值内核迁移清单：已下沉 / 迁移中 / 有意留 Python，逐项附理由与 issue。
- [hjb-subsystem](hjb-subsystem.md) — HJB 子系统目标形态：两级分工、Hamiltonian 接缝、维度上限、绑定入口、验证分层。
- [hjb-hamiltonian-dataflow](hjb-hamiltonian-dataflow.md) — 星历力模型 Hamiltonian 的配套调研：力模型与 EphemCache 现状、一次求解的数据流。

## 架构决策记录（ADR）

每篇 ADR 是一个决策快照，记录背景、决策、理由与结果；决策变化时不改写原文，而是追加修订或另写新篇。ADR 存于仓库 `docs/adr/`，面向开发协作，不进用户文档站点。索引与状态词汇表见 [`docs/adr/README.md`](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/README.md)。

影响面最大、最值得先读的几篇：

| ADR | 主题 | 与本文的关系 |
|---|---|---|
| [0011](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0011-five-layer-architecture.md) | 五层架构与全量重命名 | 分层词汇的来源 |
| [0012](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0012-dependency-direction.md) | 依赖方向规则与 CI 检查 | 模块间允许谁依赖谁 |
| [0002](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0002-rust-integrator-core.md) | Rust 积分器内核 | Python/Rust 接缝的起点 |
| [0016](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0016-ephem-cache-architecture.md) | EphemCache 星历缓存 | "Rust 不碰 cspice" 的落法 |
| [0014](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0014-api-facade-mcp-cli.md) | Facade / MCP / CLI 同源 | 任务级接口模型 |
| [0024](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0024-unified-algorithm-result-status.md) | 统一算法结果状态契约 | 结果解释层的统一词汇 |
| [0031](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0031-orbit-catalog.md) | 轨道库 catalog | 数据管理模块的落法 |

## 已知待整事项

架构不是完成时。当前登记在册的审查项：

- 时空系统三条帧转换路径并存、时间转换责任链待理清。
- Python 编排层打靶双路径、`algorithm/transfer` 数值残留待收拢。
- NLP、NSGA-II 等优化内核仍在 Python，按 [numerics-migration-status](numerics-migration-status.md) 的节奏逐步下沉。
