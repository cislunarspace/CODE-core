---
title: Differential Correction Activity Diagram / 微分修正活动图
---

# Differential Correction Activity Diagram / 微分修正活动图

[English](#differential-correction-activity-diagram) | [简体中文](#微分修正活动图)

## English

```mermaid
flowchart TD
    start([Start correction])
    config[Load CorrectionConfig strategy]
    propagate[Propagate half period (with_stm=True)]
    error[Compute constraint error vector]
    check[Converged?]
    update[Newton-update free variables]
    end([Return converged orbit])
    start --> config
    config --> propagate
    propagate --> error
    error --> check
    update --> propagate
    check -->|Yes (error < tol)| end
    check -->|No| update
```

## 微分修正活动图

```mermaid
flowchart TD
    start([开始修正])
    config[加载 CorrectionConfig 策略]
    propagate[传播半周期 (with_stm=True)]
    error[计算约束误差向量]
    check[收敛?]
    update[Newton 更新自由变量]
    end([返回收敛轨道])
    start --> config
    config --> propagate
    propagate --> error
    error --> check
    update --> propagate
    check -->|是 (error < tol)| end
    check -->|否| update
```
