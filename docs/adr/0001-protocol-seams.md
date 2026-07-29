# ADR 0001：撤回 Protocol 接缝

**状态**：已采纳
**日期**：2026-05-10
**关联 Issue**：#27

## 背景

`e2m2e/mbse/architecture/ports.py` 定义了 7 个带 `@runtime_checkable` 的 `Protocol` 类（`SystemModel`、`EOMProvider`、`Propagator`、`OrbitContainer`、`CorrectorStrategy`、`Optimizer`、`Visualizer`）。但生产代码里**一处也没用**这些 `Protocol` 类型标注——`algorithms/`、`transfer/`、`visualization/` 标注的都是具体类型（`CR3BP_Dynamics` 等）。唯一的 `isinstance(..., Protocol)` 调用只出现在 `tests/mbse/test_protocol_conformance.py`。

与此同时，`Dynamics` 基类已经通过模板方法模式（`propagate()` + `_get_eom_func()` 钩子）提供了真正的多态机制。`CR3BP_Dynamics` 和 `EphemerisDynamics` 都继承自它，`Protocol` 因此成了一层多余的并行接口。

`TransferSearch` 也明确无法使用 `Propagator`，因为它要用 `dynamics.system.mu`，而 `Protocol` 并未定义这一属性——说明这些 `Protocol` 定义对真实使用而言是不完整的。

## 决策

**撤回**：删除 `ports.py` 与 `test_protocol_conformance.py`。接受一个单态代码库——`Dynamics` 基类是唯一的多态机制。

## 理由

1. **`Protocol` 与 `Dynamics` 基类重叠。** `Dynamics` 已经定义了 `propagate()` 和 `equations_of_motion()`；`Protocol` 加了一套重复的契约，且没有运行时强制。

2. **生产代码零使用。** 库里没有任何函数签名接收 `Protocol` 类型。这 7 个 `Protocol` 纯属装饰。

3. **定义不完整。** `TransferSearch` 需要 `dynamics.system.mu`——`Propagator` 里没有。要让 `Protocol` 可用就得扩充它们，为当前不存在的收益增加复杂度。

4. **下游 issue 没有 `Protocol` 也照样推进。** Issue #31（DC → `dynamics.propagate()`）将标注为 `Dynamics` 基类；Issue #34（Transfer 合并）直接合并；#32 与 #33 不受影响。

5. **MBSE 元数据是装饰性的。** 组件注册表与需求文件以字符串形式引用 `Protocol` 名——只记录意图，不影响运行时。

## 结果

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
