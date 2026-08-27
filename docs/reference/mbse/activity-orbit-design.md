---
title: Orbit Design Activity Diagram / 轨道设计活动图
---

# Orbit Design Activity Diagram / 轨道设计活动图

[English](#orbit-design-activity-diagram) | [简体中文](#轨道设计活动图)

## English

```mermaid
flowchart TD
    start([Start])
    sys[Create CR3BP_System]
    dyn[Create CR3BP_Dynamics]
    prop[Propagate initial guess]
    correct[Differential correction]
    cont[Orbit continuation]
    end([Obtain orbit family])
    start --> sys
    sys --> dyn
    dyn --> prop
    prop --> correct
    correct --> cont
    cont --> end
```

## 轨道设计活动图

```mermaid
flowchart TD
    start([开始])
    sys[创建 CR3BP_System]
    dyn[创建 CR3BP_Dynamics]
    prop[传播初始猜测]
    correct[微分修正]
    cont[轨道延续]
    end([获得轨道族])
    start --> sys
    sys --> dyn
    dyn --> prop
    prop --> correct
    correct --> cont
    cont --> end
```
