---
title: Convergence State Machine / 收敛状态机
---

# Convergence State Machine / 收敛状态机

[English](#convergence-state-machine) | [简体中文](#收敛状态机)

## English

```mermaid
stateDiagram-v2
    [*] --> iterating
    iterating --> converged : error < tol
    iterating --> diverged : error > max
    iterating --> max_iterations : iter >= max_iter
    converged --> [*]
    diverged --> [*]
```

## 收敛状态机

```mermaid
stateDiagram-v2
    [*] --> iterating
    iterating --> converged : error < tol
    iterating --> diverged : error > max
    iterating --> max_iterations : iter >= max_iter
    converged --> [*]
    diverged --> [*]
```
