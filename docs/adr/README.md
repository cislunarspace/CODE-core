# Architecture Decision Records (ADR) / 架构决策记录（ADR）

[English](#english) | [简体中文](#简体中文)

## English

This directory records e2m2e's architecture decisions. Each ADR is a decision
snapshot: it captures the context, decision, rationale, and consequences at
the time of writing. When a decision later changes, do not rewrite the
original text; instead append a revision subsection at the end, or write a new
ADR and mark the supersession in the old one.

### Status vocabulary

- **Adopted**: the decision is in effect. Parenthetical notes may indicate
  implementation progress or partial revision, e.g.: Adopted (partially
  implemented: …), Adopted (decision 3 revised by ADR 0024).
- **Rejected**: the proposal was not adopted. The body keeps the proposal and
  the rejection rationale.
- **Superseded**: the decision was wholly overturned by a later ADR; the
  status line names the successor, e.g.: Superseded (see ADR 0024). The
  original entry is kept, never deleted.

Status describes the fate of the decision itself. When the decision's object
is to veto some mechanism (e.g. ADR 0008 vetoes runtime freezing), the status
is still Adopted, with the vetoed object noted in parentheses.

When a decision is partially revised by a later ADR, both entries keep mutual
pointers: the new ADR states in its "Related" section and relevant clauses
which clauses were revised; the old ADR gets revision notes at the revised
spots. Silent overrides without pointers violate the ADR conflict-annotation
convention.

### Numbering rules

- Numbers are four digits, increasing, never reused; normally in time order.
- Back-filled historical decisions occupy vacated numbers of their era, with
  the actual decision date noted at the top (see ADR 0005).

### Template

```markdown
# ADR XXXX: Title

**Status**: see vocabulary above
**Date**: YYYY-MM-DD
**Related Issue**: #nnn
**Related**: ADR YYYY (relationship to this entry)

## Context

Why this decision must be made now. State facts and constraints clearly,
without piling up detail.

## Decision

Itemized list; each item actionable and verifiable.

## Rationale

For each decision item, why this shape and not another. Where alternatives
exist, state why they were excluded.

## Consequences

Added / changed / unchanged. Where there is a cost, state it.
```

Optional subsections: `Alternatives compared`, `Trade-offs`,
`Revision (date, reference)`. Revision subsections are appended at the end;
original text untouched. ADRs leave no TODOs: to-dos move to issues or new
ADRs.

### Index

| No. | Title | Status |
|---|---|---|
| 0001 | Withdraw Protocol seams | Adopted |
| 0002 | Rust integrator core, Python-controlled dynamics | Adopted (with multiple revisions) |
| 0003 | Axes, ITRF93 defaults, GMAT-compatible Earth orientation | Adopted |
| 0004 | ForceModel config-driven | Adopted |
| 0005 | TwoLevelMultipleShooting as an independent algorithm | Adopted (revoked 2026-08-13: implementation deleted, see revision at end) |
| 0006 | Unified ephemeris-correction seam with registry dispatch | Adopted (revoked 2026-08-13: implementation deleted, see revision at end) |
| 0007 | Dynamic-axes state injection scheme | Adopted |
| 0008 | Revoke runtime freezing of Axes / Origin / CoordinateSystem | Adopted (freezing mechanism rejected and reverted) |
| 0009 | Enable spice feature for release wheels | Adopted (implemented) |
| 0010 | r2s2 integration and TDT+GCRS ↔ TDB+EBCRS spacetime conversion | Adopted (implemented) |
| 0011 | Five-layer architecture and radical full renaming | Adopted (implemented) |
| 0012 | Dependency-direction rules with CI import checks | Adopted (implemented) |
| 0013 | Verification strategy: complete tasks by definition | Adopted (test-tiering clause superseded by ADR 0021) |
| 0014 | Interface layer Facade/MCP/CLI | Adopted (partially implemented: Facade done, MCP/CLI placeholders) |
| 0015 | NominalOrbit contract and coordinate-conversion abstraction | Adopted (implemented) |
| 0016 | EphemCache ephemeris cache architecture | Adopted |
| 0017 | Transfer grid search: purely numerical kernel pushed down to Rayon | Adopted |
| 0018 | Jacobian interface extended with ∂a/∂v; STM covers velocity dependence | Adopted |
| 0019 | Drag Rust port uses ITRF93 pxform frame rotation (replacing ITRFApproxAxes) | Adopted |
| 0020 | Failure policy: deterministic failures raise, infeasible searches return flags, no implicit degradation | Adopted (decision 3 revised by ADR 0024) |
| 0021 | Test suite organized by functional categories; speed tiering abolished | Adopted |
| 0022 | Independent physical constants management | Adopted |
| 0023 | SciPy propagation exception for explicit event inputs | Adopted |
| 0024 | Unified algorithm result status contract | Adopted |
| 0025 | Test suite convergence: external references removed, primary marker invariant, explicit backend selection | Adopted |
| 0026 | Test suite layer clarification: coordinate ownership, forces test merge, dead-reference cleanup | Adopted |
| 0027 | System/Dynamics separation retained: dynamics directory unsplit, two classes unmerged | Adopted |
| 0028 | Planar triangular libration point family via full-period pseudo-arclength continuation | Adopted (#428 seam revised by ADR 0029) |
| 0029 | Orbit family generation via unified Rust deep module | Adopted (implemented) |
| 0030 | algorithm/forces stays at algorithm layer: Python config/orchestration surface, numerics in crates | Adopted |
| 0031 | Orbit catalog: record format, storage layout, query interface | Adopted |
| 0032 | HJB dynamics in a new crate plus binding-layer generic entry | Adopted |
| 0033 | HJB low-thrust toolchain: value-function product contract and online query interface | Adopted |
| 0034 | Scope of the ephemeris force-model Hamiltonian | Adopted |
| 0035 | GUI sidecar stdio protocol: shared Facade envelope, large arrays over binary frames | Adopted |
| 0036 | CR3BP baseline orbit-family dataset: precomputed full-family data shipped with the package | Adopted |

## 简体中文

本目录记录 e2m2e 的架构决策。每篇 ADR 是一个决策快照：写下当时的背景、决策、理由与结果。决策后来变化时，不改写原文，而是在文末追加修订小节，或另写新 ADR 并在旧篇标注取代关系。

### 状态词汇

- **已采纳**：决策生效。可用括号注明落实进度或局部修订，如：已采纳（部分实施：……）、已采纳（决策 3 经 ADR 0024 修订）。
- **已拒绝**：提议未被采纳。正文保留提议内容与拒绝理由。
- **已被取代**：决策整体被后续 ADR 推翻，状态行注明取代者，如：已被取代（见 ADR 0024）。原篇保留不删。

状态描述的是本篇决策本身的命运。若决策的对象是否决某个机制（如 ADR 0008 否决运行时冻结），状态仍是已采纳，否决对象写在括号里。

决策被后续 ADR 局部修订时，两篇互留指针：新篇在关联一节与相关条款处写明修订了哪条；旧篇在被修订处加修订注记。不写指针的静默覆盖视为违反 ADR 冲突标注约定。

### 编号规则

- 编号四位、递增、不复用，一般与时间序一致。
- 后补的历史决策占用当时空出的编号，篇首以编号说明注明实际决策时间（见 ADR 0005）。

### 模板

```markdown
# ADR XXXX：标题

**状态**：见上文词汇表
**日期**：YYYY-MM-DD
**关联 Issue**：#nnn
**关联**：ADR YYYY（与本篇的关系）

## 背景

为什么现在必须做这个决定。写清事实与约束，不堆细节。

## 决策

逐条列出，每条可执行、可检验。

## 理由

每条决策为什么是它而不是别的形状。有反方案的，写明反方案为何被排除。

## 结果

新增 / 变更 / 不变。有代价的，写明代价。
```

可选小节：`方案对比`、`取舍`、`修订（日期，关联）`。修订小节追加在文末，原文不动。ADR 不留 TODO：待办事项转 issue 或新 ADR。

### 索引

| 编号 | 标题 | 状态 |
|---|---|---|
| 0001 | 撤回 Protocol 接缝 | 已采纳 |
| 0002 | Rust 积分器内核，由 Python 控制动力学 | 已采纳（含多段修订） |
| 0003 | 坐标轴、ITRF93 默认值与 GMAT 兼容的地球定向 | 已采纳 |
| 0004 | ForceModel 配置驱动 | 已采纳 |
| 0005 | TwoLevelMultipleShooting 作为独立算法 | 已采纳（2026-08-13 撤销：实现已删除，见篇末修订） |
| 0006 | 星历修正统一接缝与注册表分发 | 已采纳（2026-08-13 撤销：实现已删除，见篇末修订） |
| 0007 | 动态坐标轴状态注入方案 | 已采纳 |
| 0008 | 撤销 Axes / Origin / CoordinateSystem 运行时冻结 | 已采纳（冻结机制被拒绝并回退） |
| 0009 | release wheel 启用 spice feature | 已采纳（已实施） |
| 0010 | r2s2 接入与 TDT+GCRS ↔ TDB+EBCRS 时空坐标转换 | 已采纳（已实施） |
| 0011 | 五层架构与激进式全量重命名 | 已采纳（已实施） |
| 0012 | 依赖方向规则与 CI import 检查 | 已采纳（已实施） |
| 0013 | 验证策略：按定义完成任务 | 已采纳（测试分层条款已被 ADR 0021 取代） |
| 0014 | 接口层 Facade/MCP/CLI | 已采纳（部分实施：Facade 已落地，MCP/CLI 占位） |
| 0015 | NominalOrbit 名义轨道契约与坐标转换抽象 | 已采纳（已实施） |
| 0016 | EphemCache 星历缓存架构 | 已采纳 |
| 0017 | 转移网格搜索纯数值内核下沉 Rayon | 已采纳 |
| 0018 | Jacobian 接口扩 ∂a/∂v，状态转移矩阵纳入速度依赖 | 已采纳 |
| 0019 | drag Rust 移植用 ITRF93 pxform 帧旋转（替 ITRFApproxAxes） | 已采纳 |
| 0020 | 失败处理策略：确定性失败抛异常，搜索不可行带标记，禁止隐式降级 | 已采纳（决策 3 经 ADR 0024 修订） |
| 0021 | 测试套件按功能类目组织，废除速度分层 | 已采纳 |
| 0022 | 物理常数独立管理 | 已采纳 |
| 0023 | 显式事件输入的 SciPy 传播例外 | 已采纳 |
| 0024 | 统一算法结果状态契约 | 已采纳 |
| 0025 | 测试套件收敛：外部参照清除、主标记守恒与后端显式选择 | 已采纳 |
| 0026 | 测试套件层级澄清：coordinate 归属、forces 测试合并与死引用清理 | 已采纳 |
| 0027 | System/Dynamics 分离保留：dynamics 目录不拆分、两类不合并 | 已采纳 |
| 0028 | 平面三角平动点族采用全周期伪弧长延拓 | 已采纳（#428 接缝经 ADR 0029 修订） |
| 0029 | 轨道族生成采用统一 Rust 深模块 | 已采纳（已实施） |
| 0030 | algorithm/forces 留在 algorithm 层：Python 配置/编排面，数值在 crates | 已采纳 |
| 0031 | 轨道库 catalog：记录格式、存储布局与查询接口 | 已采纳 |
| 0032 | HJB 动力学归属新 crate 与绑定层通用入口 | 已采纳 |
| 0033 | HJB 小推力工具链：值函数产品契约与在线查询接口 | 已采纳 |
| 0034 | 星历力模型 Hamiltonian 的范围 | 已采纳 |
| 0035 | GUI sidecar stdio 协议：共享 Facade 信封，大数组走二进制帧 | 已采纳 |
| 0036 | CR3BP 基线轨道族数据集：随包分发的预计算整族数据 | 已采纳 |
