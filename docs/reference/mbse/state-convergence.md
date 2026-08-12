---
title: 收敛状态机
---

# 收敛状态机

```mermaid
stateDiagram-v2
    [*] --> iterating
    iterating --> converged : error < tol
    iterating --> diverged : error > max
    iterating --> max_iterations : iter >= max_iter
    converged --> [*]
    diverged --> [*]
```
