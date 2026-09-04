# 架构决策记录（ADR）

本目录记录 e2m2e 的架构决策。每篇 ADR 是决策快照：写下当时的背景、
决策、理由与后果。决策日后变更时，不改写原文，而是在文末追加修订小节
或新开 ADR 并在旧文中标注取代关系。

> 语言说明：ADR 0043 起以中文书写。0001–0042 为英文历史存档——不再
> 翻译、不再维护，git 历史即全文。

## 状态词汇

- **已采纳**：决策生效。括号内可注明实现进度或部分修订，如
  已采纳（部分实现：…）、已采纳（决策 3 经 ADR 0024 修订）。
- **否决**：提案未获采纳。正文保留提案与否决理由。
- **被取代**：决策被后续 ADR 整体推翻；状态行注明后继（如
  被取代（见 ADR 0024））。原文保留，永不删除。

状态描述的是决策本身的命运。决策对象是"否决某机制"的条目（如
ADR 0008 否决运行时冻结），状态仍是已采纳，括注被否决的对象。

后续 ADR 部分修订早前决策时，两篇互指：新 ADR 在"相关"与相应条款写明
修订了哪些条款；旧 ADR 在被修订处加修订注。无指针的静默覆盖违反本目录
的冲突标注惯例。

## 编号规则

- 编号四位、递增、永不复用；一般按时间顺序。
- 回填的历史决策占用其年代的空号，文首注明实际决策日期（见 ADR 0005）。

## 索引

| 编号 | 标题 | 状态 |
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
| 0012 | Dependency-direction rules with CI import checks | Adopted (implemented; dependency table and enforcement scope revised by ADR 0039) |
| 0013 | Verification strategy: complete tasks by definition | Adopted (test-tiering clause superseded by ADR 0021) |
| 0014 | Interface layer Facade/MCP/CLI | Adopted (implemented; decisions 2 and 5 revised by ADR 0043, decision 8 completed for catalog value sets by ADR 0044) |
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
| 0031 | Orbit catalog: record format, storage layout, query interface | Adopted (decision 4 overturned by ADR 0045; decisions 1, 2, 5 revised by ADR 0045; decision 7 revised by ADR 0043) |
| 0032 | HJB dynamics in a new crate plus binding-layer generic entry | Adopted |
| 0033 | HJB low-thrust toolchain: value-function product contract and online query interface | Adopted |
| 0034 | Scope of the ephemeris force-model Hamiltonian | Adopted |
| 0035 | GUI sidecar stdio protocol: shared Facade envelope, large arrays over binary frames | Adopted |
| 0036 | CR3BP baseline orbit-family dataset: precomputed full-family data shipped with the package | Adopted |
| 0037 | Test suite time budget, minimal real-call coverage, and e2e test boundaries | Adopted |
| 0038 | IAS15 integrator and force-model parametric variational equations (ASSIST-derived); MERCURIUS not adopted | Adopted |
| 0039 | Shared-kernel leaf modules at the package root | Adopted (implemented) |
| 0040 | transfer_design converged trajectory: unified synodic-frame contract with trajectory_times | Adopted (implemented) |
| 0041 | spatiography — cislunar partition (Primer) analytic core: five-province taxonomy, [primer] constants, scales/classify/boundaries tools | Adopted (implemented) |
| 0042 | orbit taxonomy — 42-label classification of CR3BP periodic orbits: STK CODE vocabulary, self-defined analytic criteria, ingest stamping and response enrichment | Adopted (implemented; decision 5 tool-count clause superseded by ADR 0043/0044; label table relocated by ADR 0044) |
| 0043 | 接口类分家——Facade 只留任务级方法，轨道库与 spatiography 各自成类 | 已采纳（已实现） |
| 0044 | 术语清单暴露——闭值集经单一注册工具出库 | 已采纳（已实现） |
| 0045 | 轨道记录粒度——一轨一记录，族为标签 | 已采纳（已实现） |
| 0046 | 贡献流程——模板、type/kind/area 标签与 Project 流水线 | 已采纳（已实施） |
