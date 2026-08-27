# ADR 0024: Unified algorithm result status contract / 统一算法结果状态契约

[English](#adr-0024-unified-algorithm-result-status-contract) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-11
**Related**: ADR 0011 (five-layer architecture), ADR 0014 (interface Facade),
ADR 0020 (failure policy — revises its decision 3), ADR 0023 (explicit-event
SciPy exception), Issue #351

### Context

e2m2e's algorithm results once expressed outcomes via `success`, `converged`,
`correction_success`, `None`, free strings, and solver-instance state. These
conflate solver termination, candidate feasibility, and domain facts — already
causing collision candidates to be visualized as valid solutions.

The data layer's `Orbit` is an orbit container; algorithm and task-execution
outcomes belong on algorithm-layer result objects. The Facade is the task-level
public boundary and needs to translate these outcomes in a stable, machine-
decidable way.

### Decision

All e2m2e algorithm and task results uniformly carry `status: ConvergenceState`,
`cause: FailureCause`, `message: str`. `status` is the final outcome; `cause`
is a stable reason code; `message` only supplements human-readable context.
Success is fixed as `CONVERGED/NONE`; the `FailureCause`→`ConvergenceState`
mapping is unique and validated at construction. Synchronous calls never end at
`ITERATING`.

`ConvergenceState` gains `INFEASIBLE`, `COLLISION`, `FAILED`. Result objects
declare their triple and payload explicitly per domain, with no generic base
class or generic wrapper. Differential correction, continuation, and transfer
grid candidates each get concrete result objects. Non-success results may keep
approximate trajectories, candidates, or partial families but are never marked
successful.

`Orbit`, `OrbitFamily`, `TransferArc`, `EphemerisTable` keep domain-data
responsibilities. `Orbit` drops correction-process fields, retaining orbit-
geometric `closure_error`. Domain-fact booleans like `safe`, `is_periodic`,
`collision_found` fall outside this migration and remain.

Hard failures keep raising domain exceptions to cut control flow, with
exceptions also carrying the status triple + diagnostics. Soft failures return
flagged result objects. Facade responses directly contain the status triple;
"request processed successfully but scientific task infeasible" is soft failure
inside a normal response. Optional stages use `StageRecord` for applicability
and execution status — never encoding "not applicable"/"not run" as failure
causes.

Remove boolean result interfaces (`success`, `converged`, `correction_success`)
without a runtime compatibility layer. New persistence formats write only the
status triple; reading old formats fails with migration hints.

**This revises ADR 0020 decision 3**: that decision permitted boolean fields as
compatibly-retained derived properties; this one tightens to removal outright.
Beyond ADR 0020's `INFEASIBLE`/`COLLISION`, `ConvergenceState` also gains
`FAILED`.

### Trade-offs

Keeping compatibility projections of legacy booleans would reduce short-term
churn but allow state forks to regrow — rejected. Wrapping all results in a
uniform generic would mask structural differences among orbits, candidates,
families, and mission products — so unify the contract only, not the data
tiers. Splitting candidate collisions vs solver divergence into multiple status
systems would force downstream multi-way branching — both are expressed through
one final-status plus fine-grained reason codes.

### Consequences

Algorithm layer, Facade, tests, and visualization migrate to the status triple.
Python/Rust and third-party numeric libraries' native returns get translated at
the algorithm boundary without changing their external interfaces. This is a
breaking migration of public API and result persistence formats.

## 中文

**状态**：已采纳
**日期**：2026-08-11
**关联**：ADR 0011（五层架构）、ADR 0014（接口层 Facade）、ADR 0020（失败处理策略，本篇修订其决策 3）、ADR 0023（显式事件输入的 SciPy 传播例外）、Issue #351

### 背景

e2m2e 的算法结果曾以 `success`、`converged`、`correction_success`、`None`、自由字符串和 solver 实例状态表达结局。它们混淆了求解终止、候选可行性与领域事实，且已造成碰撞候选被可视化当作有效解。

数据层的 `Orbit` 是轨道容器；算法和任务执行的结局应由算法层结果对象承载。Facade 是任务级公开边界，需要能以稳定、机器可判定的方式翻译这些结局。

### 决策

所有 e2m2e 算法和任务结果统一使用 `status: ConvergenceState`、`cause: FailureCause` 和 `message: str`。`status` 表示最终结局，`cause` 表示稳定原因码，`message` 只补充人类可读的上下文。成功固定为 `CONVERGED/NONE`；`FailureCause` 到 `ConvergenceState` 的映射唯一，并在结果构造时校验。同步调用不得以 `ITERATING` 作为最终状态。

`ConvergenceState` 增加 `INFEASIBLE`、`COLLISION`、`FAILED`。结果对象按各自领域显式声明三元组和有效载荷，不引入通用基类或泛型包装器。微分修正、延拓与转移网格候选分别建立具体结果对象。非成功结果可保留近似轨迹、候选或部分轨道族，但不得标为成功。

`Orbit`、`OrbitFamily`、`TransferArc`、`EphemerisTable` 保持领域数据职责。`Orbit` 删除修正过程字段，保留轨道几何属性 `closure_error`。`safe`、`is_periodic`、`collision_found` 等领域事实布尔值不属于此次迁移，继续保留。

硬失败继续使用领域异常阻断控制流，但异常也携带状态三元组和诊断。软失败返回带状态的结果对象。Facade 响应直接含状态三元组；请求已成功处理但科学任务不可行属于正常响应中的软失败。可选阶段用 `StageRecord` 表达适用性和执行情况，不将不适用或未执行编码为失败原因。

移除 `success`、`converged`、`correction_success` 等算法结果布尔接口，不设运行时兼容层。新持久化格式只写状态三元组；旧格式读取失败并提示迁移。

**本条修订 ADR 0020 决策 3**：该决策允许布尔字段作为派生属性兼容保留，本篇收紧为一律移除；`ConvergenceState` 在 ADR 0020 规定的 `INFEASIBLE`/`COLLISION` 之外另增 `FAILED`。

### 取舍

保留多个旧布尔字段的兼容投影可以降低短期改动量，但会允许状态分叉再次出现，因此拒绝。将所有结果包装为统一泛型会掩盖轨道、候选、轨道族与任务产物的不同结构，因此只统一契约，不统一数据层次。将候选碰撞与求解发散分成多套状态体系会迫使下游维持多重判断，因此均通过统一最终状态和细粒度原因码表达。

### 结果

算法层、Facade、测试和可视化必须迁移至状态三元组。Python/Rust 与第三方数值库的原生返回值在算法边界翻译，不改变其外部接口。此次变更是公开 API 与结果持久化格式的破坏性迁移。
