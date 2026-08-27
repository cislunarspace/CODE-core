---
title: Orbit Lifecycle State Machine / 轨道生命周期状态机
---

# Orbit Lifecycle State Machine / 轨道生命周期状态机

[English](#orbit-lifecycle-state-machine) | [简体中文](#轨道生命周期状态机)

## English

```mermaid
stateDiagram-v2
    [*] --> created
    created --> properties_computed : compute_basic_properties()
    properties_computed --> stability_computed : compute_stability_index()
    stability_computed --> serialized : save_to_file()
```

## 轨道生命周期状态机

```mermaid
stateDiagram-v2
    [*] --> created
    created --> properties_computed : compute_basic_properties()
    properties_computed --> stability_computed : compute_stability_index()
    stability_computed --> serialized : save_to_file()
```
