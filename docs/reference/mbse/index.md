---
title: MBSE Models / MBSE 模型
---

# MBSE Models / MBSE 模型

[English](#english) | [简体中文](#中文)

## English

e2m2e's Model-Based Systems Engineering (MBSE) model is built around component
registration, requirement traceability, Pydantic data models, and Mermaid
diagram generation. ADR 0001 withdrew the decorative Protocol seams; the current
polymorphism seam rests on the `Dynamics` base class.

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

### Architecture

| Document | Description |
|------|------|
| [System overview](system-overview) | Block definition diagrams (BDD) & component architecture |
| [BDD: data layer](generated/bdd-data) | Data containers & kernel management components |
| [BDD: numerical layer](generated/bdd-numerical) | Rust numerical-computation facade |
| [BDD: algorithm layer](generated/bdd-algorithm) | Dynamics, correction & continuation components |
| [BDD: interface layer](generated/bdd-api) | Facade, CLI & MCP interfaces |
| [BDD: tools layer](generated/bdd-tools) | Auxiliary tools such as logging |
| [Functional requirements](generated/requirements) | Requirement registry with code traceability |

### Activity diagrams

| Document | Description |
|------|------|
| [Orbit design activity](activity-orbit-design) | End-to-end orbit design workflow |
| [Differential correction activity](activity-differential-correction) | Correction-iteration lifecycle |

### Sequence diagrams

| Document | Description |
|------|------|
| [Propagation sequence](sequence-propagation) | State-propagation call chain |
| [Correction sequence](sequence-correction) | Differential-correction message flow |

### State machines

| Document | Description |
|------|------|
| [Orbit lifecycle](state-orbit-lifecycle) | Orbit state transitions |
| [Convergence states](state-convergence) | Correction convergence states |

### Traceability

| Document | Description |
|------|------|
| [Traceability matrix](generated/traceability-matrix) | Requirements ↔ code & tests mapping |

## 中文

e2m2e 基于模型的系统工程（Model-Based Systems Engineering, MBSE）模型，围绕组件登记、需求追溯、Pydantic 数据模型和 Mermaid 图表生成构建。ADR-0001 已撤销装饰性的 Protocol 接缝；当前多态接缝由 `Dynamics` 基类承担。

### 架构

| 文档 | 说明 |
|------|------|
| [系统总览](system-overview) | 块定义图（BDD）与组件架构 |
| [BDD：数据层](generated/bdd-data) | 数据容器与内核管理组件 |
| [BDD：数值层](generated/bdd-numerical) | Rust 数值计算门面 |
| [BDD：算法层](generated/bdd-algorithm) | 动力学、修正与延拓组件 |
| [BDD：接口层](generated/bdd-api) | Facade、CLI 与 MCP 接口 |
| [BDD：工具层](generated/bdd-tools) | 日志等辅助工具 |
| [功能需求](generated/requirements) | 功能需求登记与代码追溯 |

### 活动图

| 文档 | 说明 |
|------|------|
| [轨道设计活动](activity-orbit-design) | 端到端轨道设计工作流 |
| [微分修正活动](activity-differential-correction) | 修正迭代生命周期 |

### 序列图

| 文档 | 说明 |
|------|------|
| [传播序列](sequence-propagation) | 状态传播调用链 |
| [修正序列](sequence-correction) | 微分修正消息流 |

### 状态机

| 文档 | 说明 |
|------|------|
| [轨道生命周期](state-orbit-lifecycle) | 轨道状态转换 |
| [收敛状态](state-convergence) | 修正收敛状态 |

### 追溯

| 文档 | 说明 |
|------|------|
| [追溯矩阵](generated/traceability-matrix) | 需求到代码与测试的映射 |
