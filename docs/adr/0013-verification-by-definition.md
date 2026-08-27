# ADR 0013: Verification strategy — complete tasks by definition / 验证策略：按定义完成任务，不用黄金样本、不与其他软件对拍

[English](#adr-0013-verification-strategy--complete-tasks-by-definition-no-golden-files-no-cross-software-comparison) | [简体中文](#中文)

## English

**Status**: Adopted (implemented; test-tiering clause superseded by ADR 0021)
**Date**: 2026-07-31
**Related**: ADR 0011 (five-layer architecture)

### Context

Numerical-library verification involves two distinct questions: ① are the
numbers correct; ② did results change after a code change. Golden-file
regression comparison (against manually verified outputs) and DFH cross-
checking were once introduced as means. But golden files are someone else's
concept — essentially treating previous results as the standard, premised on
the past being right; DFH cross-checking treats another software's output as
the standard. Neither answers whether the result is correct **by definition**.

Requirement: verification means **completing tasks by definition** —
correctness adjudicated by physical definition, depending on no external
software.

### Decision

1. **Correctness is adjudicated by physical definition**: analytic-solution
   comparisons (two-body propagation closure, constant circular-orbit radius,
   Jacobi conservation, STM determinant = 1 symplectic property, Hohmann Δv
   matching theory) + physical invariants. These are definitions; a correct
   computation satisfies them naturally.
2. **Test criteria allow literature formulas/analytic values, not other
   software's runtime output.** Vallado formulas, Richardson coefficients are
   astrodynamics axioms (part of the definition); running and diffing against
   DFH etc. is another software's output and unnecessary.
3. **No golden-file comparison.**
4. **e2m2e is an independent library, never forcibly compared against other
   software.** DFH served only as development-time cross-reference (local
   manual runs diagnosing magnitudes/systematic offsets); comparison scripts
   live outside CI and release artifacts.
5. **Test tiering**: Rust units (numerics vs analytic) → Python algorithm
   units (seed shapes/control-law analytic solutions/error-model statistics)
   → integration (cross-layer chains + physical quantities) → physical
   invariants throughout. (Note: test tiering superseded by ADR 0021's
   functional-category organization.)

### Rationale

1. **Completing tasks by definition needs no external reference**: physical
   laws *are* the definition; correct computation satisfies them naturally.
2. **Golden files prove regression, not correctness**: golden files may be
   wrong themselves yet keep passing forever.
3. **Cross-software comparison introduces needless coupling**: another
   software's output does not constitute e2m2e's definition.

### Consequences

- Test assertions come from closed-form solutions, conserved quantities,
  symmetries, known constants, literature formulas.
- The golden concept is removed; existing golden-related tests/scripts leave
  with `io/`, out of the final architecture.

### Revision (2026-08-16, #426 scope decision)

Decision 4's arrangement of placing comparison scripts under `scripts/` is
withdrawn. e2m2e maintains no cross-check scripts against qiao, DFH, or other
external research pipelines; such outputs constitute no contract for the
project's operation, release, or development. Developers may investigate
differences outside the repo on their own, but that investigation should be
neither a project to-do nor a test capability.

This does not affect e2m2e's own definition-level verification. Checks relying
only on e2m2e, physical definitions, and project-shipped SPICE kernels remain
by their behavioral value; they are not recast as external cross-checks merely
because external implementations used similar methods.

## 中文

**状态**：已采纳（已实施，测试分层条款已被 ADR 0021 取代）
**日期**：2026-07-31
**关联**：ADR 0011（五层架构）

### 背景

数值库验证有两个不同问题：①数值正确吗 ②代码改动后结果变了吗。曾引入黄金样本（golden file）对照（用人工验证过的输出做回归比对）和与 DFH 对拍作为验证手段。但 golden 是他人引入的概念，本质是用以前的结果当标准，前提是以前是对的；与 DFH 对拍是用另一个软件的输出当标准，两者都不回答按定义是否正确。

需求：验证**按定义完成任务即可**：正确性由物理定义裁决，不依赖任何外部软件。

### 决策

1. **正确性由物理定义裁决**：解析解对照（二体传播闭合、圆轨道半径不变、Jacobi 常数守恒、STM 行列式=1 辛性质、霍曼转移 Δv 匹配理论值）+ 物理不变量。这些是定义，算对了自然满足。
2. **测试标准允许文献公式/解析值，不允许其他软件运行输出**。Vallado 公式、Richardson 系数是轨道力学公理（定义的一部分）；跟 DFH 等软件跑一遍比对是别的软件输出，不需要。
3. **不使用黄金样本（golden file）对照**。
4. **e2m2e 是独立库，不与其他软件强制对比**。DFH 仅作开发期交叉参考（本地手动跑，诊断量级/系统性偏差），比对脚本放 `scripts/` 不进 CI、不进发布包。
5. **测试分层**：Rust 单元（数值方法 vs 解析解）→ Python 算法单元（种子形状/控制律解析解/误差模型统计）→ 集成（跨层链路 + 物理量）→ 物理不变量贯穿（注：测试分层已由 ADR 0021 取代，改为按功能类目组织）。

### 理由

1. **按定义完成任务不需要外部参照**：物理定律就是定义，算对了自然满足。
2. **golden 只证回归不证正确**：黄金样本本身可能是错的，它会一直通过。
3. **与其他软件对拍引入不必要耦合**：另一软件的输出不构成 e2m2e 的定义。

### 结果

- 测试断言来自解析闭式解、守恒量、对称性、已知常数、文献公式。
- 移除 golden 概念；现有 golden 相关测试/脚本随 io/ 一起不入最终架构。

### 修订（2026-08-16，#426 范围决定）

决策 4 中比对脚本放 `scripts/` 的安排撤回。e2m2e 不维护与 qiao、DFH
或其他外部研究流水线的对拍脚本；这些输出不构成项目的运行、发布或开发期
契约。开发者可在仓库外自行调查差异，但不应把该调查作为项目待办或测试能力。

这不影响 e2m2e 自身的定义级验证。凡只依赖 e2m2e、物理定义和项目支持的
SPICE 内核的检查，仍按其行为价值保留；它们不因曾与外部实现使用相近方法而
被视为外部对拍。
