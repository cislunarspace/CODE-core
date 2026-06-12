---
title: MBSE 模型
---

# MBSE 模型

e2m2e 基于模型的系统工程（Model-Based Systems Engineering, MBSE）模型，围绕组件登记、需求追溯、Pydantic 数据模型和 Mermaid 图表生成构建。ADR-0001 已撤销装饰性的 Protocol 接缝；当前多态接缝由 `Dynamics` 基类承担。

## 架构

| 文档 | 说明 |
|------|------|
| [系统总览](system-overview) | 块定义图（BDD）与组件架构 |
| [BDD：核心模块](bdd-core) | 核心模块块定义 — System, Dynamics, Orbit |
| [BDD：算法模块](bdd-algorithms) | 算法模块块定义 — DifferentialCorrection, Continuation, Stability |
| [功能需求](requirements) | 功能需求登记与追溯 |

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
| [追溯矩阵](traceability-matrix) | 需求到组件的映射 |
