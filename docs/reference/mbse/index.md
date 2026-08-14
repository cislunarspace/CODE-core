---
title: MBSE 模型
---

# MBSE 模型

e2m2e 基于模型的系统工程（Model-Based Systems Engineering, MBSE）模型，围绕组件登记、需求追溯、Pydantic 数据模型和 Mermaid 图表生成构建。ADR-0001 已撤销装饰性的 Protocol 接缝；当前多态接缝由 `Dynamics` 基类承担。

```{toctree}
:hidden:

system-overview
generated/bdd-data
generated/bdd-numerical
generated/bdd-algorithm
generated/bdd-api
generated/bdd-tools
generated/requirements
activity-orbit-design
activity-differential-correction
sequence-propagation
sequence-correction
state-orbit-lifecycle
state-convergence
generated/traceability-matrix
```

## 架构

| 文档 | 说明 |
|------|------|
| [系统总览](system-overview) | 块定义图（BDD）与组件架构 |
| [BDD：数据层](generated/bdd-data) | 数据容器与内核管理组件 |
| [BDD：数值层](generated/bdd-numerical) | Rust 数值计算门面 |
| [BDD：算法层](generated/bdd-algorithm) | 动力学、修正与延拓组件 |
| [BDD：接口层](generated/bdd-api) | Facade、CLI 与 MCP 接口 |
| [BDD：工具层](generated/bdd-tools) | 日志等辅助工具 |
| [功能需求](generated/requirements) | 功能需求登记与代码追溯 |

## 活动图

| 文档 | 说明 |
|------|------|
| [轨道设计活动](activity-orbit-design) | 端到端轨道设计工作流 |
| [微分修正活动](activity-differential-correction) | 修正迭代生命周期 |

## 序列图

| 文档 | 说明 |
|------|------|
| [传播序列](sequence-propagation) | 状态传播调用链 |
| [修正序列](sequence-correction) | 微分修正消息流 |

## 状态机

| 文档 | 说明 |
|------|------|
| [轨道生命周期](state-orbit-lifecycle) | 轨道状态转换 |
| [收敛状态](state-convergence) | 修正收敛状态 |

## 追溯

| 文档 | 说明 |
|------|------|
| [追溯矩阵](generated/traceability-matrix) | 需求到代码与测试的映射 |
