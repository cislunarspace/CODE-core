# ADR 0023：显式事件输入的 SciPy 传播例外

**状态**：已采纳
**日期**：2026-08-11
**关联 Issue**：#378

## 背景

CR3BP 与 BCR4BP 的常规传播已经接入 Rust 路径。Rust 传播函数不承担这两类动力学对象的 SciPy 事件接口语义：事件函数属性、增广 STM 事件状态和终止结果需要保持现有 SciPy 契约。另一方面，Rust 扩展缺失不能被当作普通运行时降级条件，否则会掩盖构建问题。

## 决策

CR3BP 与 BCR4BP 只有在调用者显式传入 `events` 时才使用 SciPy 传播。这是由 `events` 输入触发的有意例外，与 Rust 扩展是否可用无关；扩展可用性变化不会改变该分派规则。

不传 `events` 时，传播必须要求对应 Rust 扩展符号可用。扩展缺失或符号缺失显式抛出 `RustExtensionUnavailableError`，不得回退 SciPy。ForceModel 的事件接口当前仍明确不支持，因为 compiled-forces 尚未提供事件传播 API；底层 `e2m2e.integrators.solve_ivp_events` 独立提供并测试 Rust 事件细化能力。

## 结果

显式事件调用继续获得 SciPy 的事件语义和结果字段；普通传播保持 Rust 数值路径和缺失扩展的显式失败。两种场景的分派原因可由输入参数直接判断，不再把“扩展缺失”与“事件例外”混为一谈。
