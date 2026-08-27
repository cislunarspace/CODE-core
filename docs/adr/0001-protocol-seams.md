# ADR 0001: Withdraw Protocol seams / 撤回 Protocol 接缝

[English](#adr-0001-withdraw-protocol-seams) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-05-10
**Related Issue**: #27

### Context

`e2m2e/mbse/architecture/ports.py` defines seven `@runtime_checkable`
`Protocol` classes (`SystemModel`, `EOMProvider`, `Propagator`,
`OrbitContainer`, `CorrectorStrategy`, `Optimizer`, `Visualizer`). But **not a
single place** in production code uses these `Protocol` type annotations:
`algorithms/`, `transfer/`, and `visualization/` all annotate concrete types
(`CR3BP_Dynamics`, etc.). The only `isinstance(..., Protocol)` call appears
solely in `tests/mbse/test_protocol_conformance.py`.

Meanwhile the `Dynamics` base class already provides a genuine polymorphism
mechanism via the template-method pattern (`propagate()` + `_get_eom_func()`
hook). Both `CR3BP_Dynamics` and `EphemerisDynamics` inherit from it; the
`Protocol`s therefore form a redundant parallel interface.

`TransferSearch` also demonstrably cannot use `Propagator`, because it needs
`dynamics.system.mu`, which the `Protocol` does not define — evidence that
these `Protocol` definitions are incomplete for real use anyway.

### Decision

**Withdraw**: delete `ports.py` and `test_protocol_conformance.py`. Accept a
monomorphic codebase: the `Dynamics` base class is the only polymorphism
mechanism.

### Rationale

1. **The `Protocol`s overlap the `Dynamics` base class.** `Dynamics` already
   defines `propagate()` and `equations_of_motion()`; the `Protocol`s add a
   duplicate set of contracts without runtime enforcement.

2. **Zero production usage.** No function signature anywhere in the library
   accepts a `Protocol` type. These seven `Protocol`s are pure decoration.

3. **Incomplete definitions.** `TransferSearch` needs `dynamics.system.mu`;
   `Propagator` lacks it. Making the `Protocol`s usable would mean expanding
   them — added complexity for a benefit that does not currently exist.

4. **Downstream issues proceed fine without `Protocol`s.** Issue #31 (DC →
   `dynamics.propagate()`) annotates against the `Dynamics` base class; #34
   (Transfer merge) merges directly; #32 and #33 are unaffected.

5. **MBSE metadata is decorative.** Component registries and requirement files
   reference `Protocol` names as strings, recording intent only, with no
   runtime effect.

### Consequences

#### Removed

- `e2m2e/mbse/architecture/ports.py`
- `tests/mbse/test_protocol_conformance.py`
- `Protocol` exports in `mbse/architecture/__init__.py`
- `Protocol` references in MBSE component metadata

#### Kept

- All public API classes (`CR3BP_Dynamics`, `Dynamics`, `Orbit`, `Transfer`,
  etc.)
- The `Dynamics` base class as the genuine polymorphism mechanism
- MBSE infrastructure (components, requirements, diagrams), references updated

#### Downstream impact

| Issue | Path forward |
|-------|----------|
| #31 DC → `propagate()` | Annotate against `Dynamics` base |
| #34 Transfer + Optimizer merge | Direct merge, no `Protocol` indirection |
| #32 Stability merge | Unaffected |
| #33 Visualization flattening | Unaffected |

## 中文

**状态**：已采纳
**日期**：2026-05-10
**关联 Issue**：#27

### 背景

`e2m2e/mbse/architecture/ports.py` 定义了 7 个带 `@runtime_checkable` 的 `Protocol` 类（`SystemModel`、`EOMProvider`、`Propagator`、`OrbitContainer`、`CorrectorStrategy`、`Optimizer`、`Visualizer`）。但生产代码里**一处也没用**这些 `Protocol` 类型标注，`algorithms/`、`transfer/`、`visualization/` 标注的都是具体类型（`CR3BP_Dynamics` 等）。唯一的 `isinstance(..., Protocol)` 调用只出现在 `tests/mbse/test_protocol_conformance.py`。

与此同时，`Dynamics` 基类已经通过模板方法模式（`propagate()` + `_get_eom_func()` 钩子）提供了真正的多态机制。`CR3BP_Dynamics` 和 `EphemerisDynamics` 都继承自它，`Protocol` 因此成了一层多余的并行接口。

`TransferSearch` 也明确无法使用 `Propagator`，因为它要用 `dynamics.system.mu`，而 `Protocol` 并未定义这一属性，说明这些 `Protocol` 定义对真实使用而言并不完整。

### 决策

**撤回**：删除 `ports.py` 与 `test_protocol_conformance.py`。接受一个单态代码库：`Dynamics` 基类是唯一的多态机制。

### 理由

1. **`Protocol` 与 `Dynamics` 基类重叠。** `Dynamics` 已经定义了 `propagate()` 和 `equations_of_motion()`；`Protocol` 加了一套重复的契约，且没有运行时强制。

2. **生产代码零使用。** 库里没有任何函数签名接收 `Protocol` 类型。这 7 个 `Protocol` 纯属装饰。

3. **定义不完整。** `TransferSearch` 需要 `dynamics.system.mu`，`Propagator` 里没有。要让 `Protocol` 可用就得扩充它们，为当前不存在的收益增加复杂度。

4. **下游 issue 没有 `Protocol` 也照样推进。** Issue #31（DC → `dynamics.propagate()`）将标注为 `Dynamics` 基类；Issue #34（Transfer 合并）直接合并；#32 与 #33 不受影响。

5. **MBSE 元数据是装饰性的。** 组件注册表与需求文件以字符串形式引用 `Protocol` 名，只记录意图，不影响运行时。

### 结果

### 移除

- `e2m2e/mbse/architecture/ports.py`
- `tests/mbse/test_protocol_conformance.py`
- `mbse/architecture/__init__.py` 中的 `Protocol` 导出
- MBSE 组件元数据中的 `Protocol` 引用

### 保留

- 所有公开 API 类（`CR3BP_Dynamics`、`Dynamics`、`Orbit`、`Transfer` 等）
- `Dynamics` 基类作为真正的多态机制
- MBSE 基础设施（组件、需求、图表），引用已更新

### 下游影响

| Issue | 后续路径 |
|-------|----------|
| #31 DC → `propagate()` | 标注为 `Dynamics` 基类 |
| #34 Transfer + Optimizer 合并 | 直接合并，不经 `Protocol` 间接层 |
| #32 稳定性合并 | 不受影响 |
| #33 可视化扁平化 | 不受影响 |
